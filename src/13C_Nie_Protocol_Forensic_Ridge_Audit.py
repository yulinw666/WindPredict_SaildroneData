# -*- coding: utf-8 -*-
"""
13C_Nie_Protocol_Forensic_Ridge_Audit.py

Protocol-forensic audit for the BP-STGNN / Nie same-mission benchmark.

It reports, for Ridge:
  * development TRAIN in-sample RMSE
  * development blocked-OOF RMSE
  * Tropical Atlantic TEST RMSE

Audits:
  1) exact existing Stage-12G phase-0 benchmark;
  2) UTC 10-min point-sampling phases 0..9;
  3) mission-start-anchored point sampling;
  4) 10-min-after-sampling vs 1-min-before-sampling dT/dP/dRH;
  5) current COMMON8 validity vs BP7 validity (wing not required);
  6) short-gap cubic-spline sensitivity (1/3/5 min);
  7) Atlantic-only training + Tropical chronological validation/guarded holdout.

Scientific guardrail:
Atlantic-only + Tropical validation is an alternative domain-matched diagnostic,
NOT Nie et al.'s published split. The published study uses the first three
missions for training/validation and Tropical Atlantic as the independent test.

Recommended PowerShell:
python "D:\\project\\WindPredict_SaildroneData\\src\\13C_Nie_Protocol_Forensic_Ridge_Audit.py" --project-root "D:\\project\\WindPredict_SaildroneData" --stage12g-dataset "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\12G_Nie_point_sampled_benchmark_v0_1\\dataset" --raw-dir "D:\\project\\WindPredict_SaildroneData\\data\\raw\\RawData" --output-dir "D:\\project\\WindPredict_SaildroneData\\data\\forecasting\\13C_Nie_Protocol_Forensic_Ridge_Audit_v0_1"

Smoke (does not rebuild raw variants): add --debug-fast
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.linear_model import Ridge

SCRIPT_VERSION = "0.1.1-13C-protocol-forensic"
ROOT = Path(r"D:\project\WindPredict_SaildroneData")
DEFAULT_12G = ROOT / "data" / "forecasting" / "12G_Nie_point_sampled_benchmark_v0_1" / "dataset"
DEFAULT_RAW = ROOT / "data" / "raw" / "RawData"
DEFAULT_OUT = ROOT / "data" / "forecasting" / "13C_Nie_Protocol_Forensic_Ridge_Audit_v0_1"

DEV = ["Antarctic", "Atlantic", "West Coast"]
TEST = "Tropical Atlantic"
ALL = DEV + [TEST]
ONE_MIN_NS = 60 * 1_000_000_000
TEN_MIN_NS = 10 * ONE_MIN_NS
L = 6
RIDGE_ALPHA = 1.0
OOF_FOLDS = 5
EPS = 1e-12

J_NAMES = ["U","V","T","RH","SOG","COG_sin","COG_cos","WING_ANGLE_sin","WING_ANGLE_cos"]
P_NAMES = ["U","V","SOG","COG","T","RH","P","dT","dP","dRH"]
BASE4 = ["U","V","T","RH"]
RAW_COMMON8 = ["U","V","SOG","COG","WING_ANGLE","T","RH","P"]
RAW_BP7 = ["U","V","SOG","COG","T","RH","P"]
RAW10 = ["U","V","SOG","COG","T","RH","P","dT","dP","dRH"]
RAW10_IDX = {n:i for i,n in enumerate(RAW10)}
BP = {"U":0.748640,"V":0.705060,"WS":0.699400,"WD":5.584000}


def log(s=""):
    print(s, flush=True)


def save_json(path, obj):
    def cv(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return None if not np.isfinite(x) else float(x)
        if isinstance(x, (np.bool_,)): return bool(x)
        if isinstance(x, dict): return {str(k):cv(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [cv(v) for v in x]
        return x
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cv(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def wd_from_uv(y):
    y = np.asarray(y, float)
    return np.degrees(np.arctan2(-y[:,0], -y[:,1])) % 360.0


def circdiff(a,b):
    return (np.asarray(a)-np.asarray(b)+180.0)%360.0-180.0


def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); e=p-y
    sy=np.linalg.norm(y,axis=1); sp=np.linalg.norm(p,axis=1)
    de=circdiff(wd_from_uv(p),wd_from_uv(y))
    return {
        "U_RMSE_mps":float(np.sqrt(np.mean(e[:,0]**2))),
        "V_RMSE_mps":float(np.sqrt(np.mean(e[:,1]**2))),
        "vector_RMSE_mps":float(np.sqrt(np.mean(np.sum(e**2,axis=1)))),
        "WS_RMSE_mps":float(np.sqrt(np.mean((sp-sy)**2))),
        "WD_RMSE_deg":float(np.sqrt(np.mean(de**2))),
    }


def pref(m,p): return {f"{p}_{k}":v for k,v in m.items()}


def fit_scaler(X,y):
    X=np.asarray(X,float); y=np.asarray(y,float)
    xm=X.mean((0,1)); xs=X.std((0,1)); xs=np.where(xs<1e-8,1.0,xs)
    ym=y.mean(0); ys=y.std(0); ys=np.where(ys<1e-8,1.0,ys)
    return {"xm":xm.astype(np.float32),"xs":xs.astype(np.float32),"ym":ym.astype(np.float32),"ys":ys.astype(np.float32)}


def tx(X,s): return ((np.asarray(X,np.float32)-s["xm"][None,None,:])/s["xs"][None,None,:]).astype(np.float32)
def ty(y,s): return ((np.asarray(y,np.float32)-s["ym"][None,:])/s["ys"][None,:]).astype(np.float32)
def iy(z,s): return (np.asarray(z,np.float32)*s["ys"][None,:]+s["ym"][None,:]).astype(np.float32)


def fit_ridge(X,y):
    s=fit_scaler(X,y); m=Ridge(alpha=RIDGE_ALPHA)
    m.fit(tx(X,s).reshape(len(X),-1),ty(y,s))
    return m,s


def pred_ridge(m,s,X): return iy(m.predict(tx(X,s).reshape(len(X),-1)),s)


def blocked_folds(labels):
    labels=np.asarray(labels,object); f=np.full(len(labels),-1,int)
    for mission in np.unique(labels):
        idx=np.flatnonzero(labels==mission)
        for k,ch in enumerate(np.array_split(idx,OOF_FOLDS)): f[ch]=k
    if (f<0).any(): raise RuntimeError("OOF assignment failed")
    return f


def oof_ridge(X,y,labels):
    f=blocked_folds(labels); out=np.full_like(y,np.nan,np.float32)
    for k in range(OOF_FOLDS):
        va=f==k; tr=~va; m,s=fit_ridge(X[tr],y[tr]); out[va]=pred_ridge(m,s,X[va])
    if not np.isfinite(out).all(): raise RuntimeError("Nonfinite OOF")
    return out


def evaluate_ridge(protocol, model_name, Xtr,ytr,labels,Xte,yte,persist,meta):
    m,s=fit_ridge(Xtr,ytr)
    pin=pred_ridge(m,s,Xtr); poof=oof_ridge(Xtr,ytr,labels); pte=pred_ridge(m,s,Xte)
    return {"protocol":protocol,"model":model_name,"train_samples":len(ytr),"test_samples":len(yte),**meta,
            **pref(metrics(ytr,pin),"train_in_sample"),**pref(metrics(ytr,poof),"train_blocked_OOF"),
            **pref(metrics(yte,pte),"test"),**pref(metrics(yte,persist),"test_persistence")}


# ---------------- existing Stage12G ----------------
def track_path(ds,track,mission):
    d=ds/track; safe=mission.replace(" ","_")
    c=[d/f"{safe}_TEST.npz",d/f"{safe}.npz"] if mission==TEST else [d/f"{safe}.npz"]
    for p in c:
        if p.exists(): return p
    raise FileNotFoundError(c)


def load_track(ds,track,mission):
    p=track_path(ds,track,mission)
    with np.load(p,allow_pickle=False) as z:
        X=np.asarray(z["X_raw"],np.float32); y=np.asarray(z["y_wind_raw"],np.float32)
        c=np.asarray(z["context_end_time_ns"],np.int64).reshape(-1); t=np.asarray(z["target_time_ns"],np.int64).reshape(-1)
        names=[str(x) for x in z["feature_names"].tolist()]
    if y.ndim==3:y=y[:,0,:]
    if not np.all(t-c==TEN_MIN_NS): raise RuntimeError(f"{p}: target not +10 min")
    return {"X":X,"y":y,"context":c,"target":t,"names":names,"mission":mission,"path":p}


def cat(items):
    return {"X":np.concatenate([x["X"] for x in items]),"y":np.concatenate([x["y"] for x in items]),
            "target":np.concatenate([x["target"] for x in items]),
            "labels":np.concatenate([np.asarray([x["mission"]]*len(x["y"]),object) for x in items])}


def select_names(X,names,wanted):
    return np.asarray(X[:,:,[names.index(w) for w in wanted]],np.float32)


def exact12g(ds):
    rows=[]
    js=[load_track(ds,"track_J_joint_compatible",m) for m in DEV]; jt=load_track(ds,"track_J_joint_compatible",TEST); jd=cat(js)
    Xtr=select_names(jd["X"],js[0]["names"],BASE4); Xte=select_names(jt["X"],jt["names"],BASE4)
    rows.append(evaluate_ridge("Existing12G_exact","Base4-Ridge",Xtr,jd["y"],jd["labels"],Xte,jt["y"],Xte[:,-1,:2],
                               {"phase":0,"anchor":"UTC","diff":"10min_after_sampling","spline_gap":0,"eligibility":"COMMON8"}))
    ps=[load_track(ds,"track_P_paper_informed",m) for m in DEV]; pt=load_track(ds,"track_P_paper_informed",TEST); pdv=cat(ps)
    rows.append(evaluate_ridge("Existing12G_exact","Full10-Ridge",pdv["X"],pdv["y"],pdv["labels"],pt["X"],pt["y"],pt["X"][:,-1,:2],
                               {"phase":0,"anchor":"UTC","diff":"10min_after_sampling","spline_gap":0,"eligibility":"COMMON8"}))
    return pd.DataFrame(rows),{"js":js,"jt":jt,"ps":ps,"pt":pt}


# ---------------- raw protocol rebuild ----------------
def raw_loader(project_root,raw_dir,progress_every):
    p=project_root/"src"/"12B_build_Nie_same_mission_benchmark_dataset.py"
    if not p.exists(): raise FileNotFoundError(p)
    b=load_module(p,"stage12b_for_13c"); xr=b.import_xarray(); grouped=b.discover_mission_files(raw_dir)
    out={}; audit=[]
    for mission in ALL:
        raw,ra,_=b.load_mission_raw(xr,mission,grouped[mission],progress_every)
        raw=raw.sort_values("time_ns").drop_duplicates("time_ns",keep="last").reset_index(drop=True)
        missing=[c for c in RAW_COMMON8 if c not in raw.columns]
        if missing: raise RuntimeError(f"{mission} missing {missing}")
        out[mission]=raw
        audit.append({"mission":mission,"rows":len(raw),"first":str(np.datetime64(int(raw.time_ns.iloc[0]),'ns')),
                      "last":str(np.datetime64(int(raw.time_ns.iloc[-1]),'ns')),"source_files":len(grouped[mission]),"loader_audit":str(ra)})
    return out,pd.DataFrame(audit)


def short_gap_mask(valid,maxgap):
    valid=np.asarray(valid,bool); fill=np.zeros(len(valid),bool); i=0
    while i<len(valid):
        if valid[i]: i+=1; continue
        a=i
        while i<len(valid) and not valid[i]: i+=1
        b=i
        if b-a<=maxgap and a>0 and b<len(valid) and valid[a-1] and valid[b]: fill[a:b]=True
    return fill


def fill_linear(v,maxgap):
    v=np.asarray(v,float).copy(); valid=np.isfinite(v); mask=short_gap_mask(valid,maxgap)
    if not mask.any(): return v,0
    x=np.arange(len(v),dtype=float); xf=x[valid]; yf=v[valid]
    if len(xf)>=4: v[mask]=CubicSpline(xf,yf,bc_type="natural",extrapolate=False)(x[mask])
    elif len(xf)>=2: v[mask]=np.interp(x[mask],xf,yf)
    return v,int(mask.sum())


def fill_angle(v,maxgap):
    v=np.asarray(v,float).copy(); valid=np.isfinite(v); mask=short_gap_mask(valid,maxgap)
    if not mask.any(): return v,0
    rad=np.deg2rad(v); sv=np.sin(rad); cv=np.cos(rad); sv[~valid]=np.nan; cv[~valid]=np.nan
    sf,_=fill_linear(sv,maxgap); cf,_=fill_linear(cv,maxgap); ang=np.degrees(np.arctan2(sf,cf))%360.0; v[mask]=ang[mask]
    return v,int(mask.sum())


def regularize(raw,gap):
    raw=raw.sort_values("time_ns").drop_duplicates("time_ns",keep="last")
    t0=int(raw.time_ns.iloc[0]); t1=int(raw.time_ns.iloc[-1]); full=np.arange(t0,t1+ONE_MIN_NS,ONE_MIN_NS,dtype=np.int64)
    d=raw.set_index("time_ns").reindex(full); d.index.name="time_ns"; counts={}
    if gap>0:
        for c in ["U","V","SOG","T","RH","P"]: d[c],counts[c]=fill_linear(d[c].to_numpy(float),gap)
        for c in ["COG","WING_ANGLE"]: d[c],counts[c]=fill_angle(d[c].to_numpy(float),gap)
    else: counts={c:0 for c in RAW_COMMON8}
    return d.reset_index(),counts


def build_samples(raw,mission,phase_mode="UTC",phase=0,diff_mode="10min_after_sampling",gap=0,eligibility="COMMON8"):
    d,counts=regularize(raw,gap); ti=d.time_ns.to_numpy(np.int64); mi=ti//ONE_MIN_NS
    used=int(phase) if phase_mode=="UTC" else int((int(raw.time_ns.iloc[0])//ONE_MIN_NS)%10)
    # 1-min diffs before sampling
    for src,name in [("T","dT1"),("P","dP1"),("RH","dRH1")]:
        a=d[src].to_numpy(float); z=np.full(len(a),np.nan); good=np.isfinite(a[1:])&np.isfinite(a[:-1]); q=np.full(len(a)-1,np.nan); q[good]=a[1:][good]-a[:-1][good]; z[1:]=q; d[name]=z
    pts=d.loc[(mi%10)==used].copy().reset_index(drop=True)
    needed=RAW_COMMON8 if eligibility=="COMMON8" else RAW_BP7
    finite=np.isfinite(pts[needed].to_numpy(float)).all(1); before=len(pts); pts=pts.loc[finite].copy().reset_index(drop=True)
    if diff_mode=="10min_after_sampling":
        tt=pts.time_ns.to_numpy(np.int64); cont=np.zeros(len(pts),bool); cont[1:]=np.diff(tt)==TEN_MIN_NS
        for src,name in [("T","dT"),("P","dP"),("RH","dRH")]:
            a=pts[src].to_numpy(float); z=np.full(len(a),np.nan); q=np.full(len(a)-1,np.nan); rd=a[1:]-a[:-1]; good=cont[1:]&np.isfinite(rd); q[good]=rd[good]; z[1:]=q; pts[name]=z
    elif diff_mode=="1min_before_sampling":
        pts["dT"]=pts["dT1"]; pts["dP"]=pts["dP1"]; pts["dRH"]=pts["dRH1"]
    else: raise ValueError(diff_mode)
    tt=pts.time_ns.to_numpy(np.int64); X=[]; y=[]; per=[]; ce=[]; tg=[]; rej_nc=rej_d=0
    for c in range(L-1,len(pts)-1):
        a=c-L+1; t=c+1; expected=tt[a]+np.arange(L+1,dtype=np.int64)*TEN_MIN_NS
        if not np.array_equal(tt[a:t+1],expected): rej_nc+=1; continue
        ctx=pts.iloc[a:c+1]
        if not np.isfinite(ctx[["dT","dP","dRH"]].to_numpy(float)).all(): rej_d+=1; continue
        xx=ctx[RAW10].to_numpy(float); yy=np.array([pts.U.iloc[t],pts.V.iloc[t]],float); pp=np.array([ctx.U.iloc[-1],ctx.V.iloc[-1]],float)
        if not (np.isfinite(xx).all() and np.isfinite(yy).all()): continue
        X.append(xx); y.append(yy); per.append(pp); ce.append(int(tt[c])); tg.append(int(tt[t]))
    if not X: raise RuntimeError(f"{mission}: no samples")
    X=np.asarray(X,np.float32); y=np.asarray(y,np.float32); per=np.asarray(per,np.float32)
    X4=X[:,:,[RAW10_IDX[n] for n in BASE4]]
    return {"X4":X4,"X10":X,"y":y,"persist":per,"target":np.asarray(tg,np.int64),"mission":mission,
            "audit":{"mission":mission,"used_phase":used,"phase_mode":phase_mode,"diff":diff_mode,"gap":gap,"eligibility":eligibility,
                     "selected_before_finite":before,"selected_after_finite":len(pts),"samples":len(y),"rejected_noncontiguous":rej_nc,
                     "rejected_diff":rej_d,"filled_total":int(sum(counts.values())),**{f"filled_{k}":int(v) for k,v in counts.items()}}}


def cat_raw(items,key):
    return {"X":np.concatenate([x[key] for x in items]),"y":np.concatenate([x["y"] for x in items]),
            "persist":np.concatenate([x["persist"] for x in items]),
            "labels":np.concatenate([np.asarray([x["mission"]]*len(x["y"]),object) for x in items])}


def eval_raw_protocol(name,items,meta):
    rows=[]
    for model,key in [("Base4-Ridge","X4"),("Full10-Ridge","X10")]:
        tr=cat_raw([items[m] for m in DEV],key); te=cat_raw([items[TEST]],key)
        rows.append(evaluate_ridge(name,model,tr["X"],tr["y"],tr["labels"],te["X"],te["y"],te["persist"],meta))
    return rows


# ---------------- Atlantic-only ----------------
def atlantic_only(cache):
    rows=[]
    pairs=[]
    aj=next(x for x in cache["js"] if x["mission"]=="Atlantic"); tj=cache["jt"]
    pairs.append(("Base4-Ridge",select_names(aj["X"],aj["names"],BASE4),select_names(tj["X"],tj["names"],BASE4),aj,tj))
    ap=next(x for x in cache["ps"] if x["mission"]=="Atlantic"); tp=cache["pt"]; pairs.append(("Full10-Ridge",ap["X"],tp["X"],ap,tp))
    for model,Xtr,Xte,tr,te in pairs:
        m,s=fit_ridge(Xtr,tr["y"]); pin=pred_ridge(m,s,Xtr); pfull=pred_ridge(m,s,Xte)
        order=np.argsort(te["target"]); nval=max(1,int(math.floor(0.20*len(order)))); vi=order[:nval]; last=int(te["target"][vi].max())
        hi=np.flatnonzero(te["target"]>last+6*TEN_MIN_NS)
        rows.append({"protocol":"AtlanticOnly_Existing12G_phase0","model":model,"train_samples":len(tr["y"]),"Tropical_full_samples":len(te["y"]),
                     "Tropical_validation_samples":len(vi),"Tropical_guarded_holdout_samples":len(hi),
                     **pref(metrics(tr["y"],pin),"Atlantic_train_in_sample"),**pref(metrics(te["y"],pfull),"Tropical_full_external"),
                     **pref(metrics(te["y"][vi],pfull[vi]),"Tropical_validation20"),**pref(metrics(te["y"][hi],pfull[hi]),"Tropical_guarded_holdout")})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",type=Path,default=ROOT); ap.add_argument("--stage12g-dataset",type=Path,default=DEFAULT_12G)
    ap.add_argument("--raw-dir",type=Path,default=DEFAULT_RAW); ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--progress-every",type=int,default=25); ap.add_argument("--debug-fast",action="store_true")
    a=ap.parse_args(); out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    log("="*132); log("13C — NIE PROTOCOL FORENSIC RIDGE AUDIT"); log("="*132)

    # Exact current 12G
    cur,cache=exact12g(a.stage12g_dataset); cur.to_csv(out/"13C_CURRENT12G_RIDGE_TRAIN_TEST.csv",index=False,encoding="utf-8-sig")
    log("\nCURRENT 12G — RIDGE TRAIN / OOF / TROPICAL TEST"); log(cur.to_string(index=False))

    atl=atlantic_only(cache); atl.to_csv(out/"13C_ATLANTIC_ONLY_DIAGNOSTIC.csv",index=False,encoding="utf-8-sig")
    log("\nATLANTIC-ONLY TRAIN / TROPICAL VALIDATION + GUARDED HOLDOUT"); log(atl.to_string(index=False))
    if a.debug_fast:
        log("\n[DEBUG FAST] Exact12G + Atlantic-only audit complete; raw variants skipped."); return 0

    raws,load_audit=raw_loader(a.project_root,a.raw_dir,a.progress_every); load_audit.to_csv(out/"13C_RAW_LOAD_AUDIT.csv",index=False,encoding="utf-8-sig")
    rows=[]; builds=[]; phase_rows=[]

    # phases 0..9
    for ph in range(10):
        name=f"UTCphase{ph}_D10_COMMON8"; log(f"[PHASE {ph}] {name}"); items={}
        for mission in ALL:
            x=build_samples(raws[mission],mission,"UTC",ph,"10min_after_sampling",0,"COMMON8"); items[mission]=x; builds.append({"protocol":name,**x["audit"]})
        rr=eval_raw_protocol(name,items,{"phase":ph,"anchor":"UTC","diff":"10min_after_sampling","spline_gap":0,"eligibility":"COMMON8"}); rows+=rr; phase_rows+=rr
        b=next(r for r in rr if r["model"]=="Base4-Ridge"); log(f"  Base4 Tropical U/V/WD = {b['test_U_RMSE_mps']:.6f} / {b['test_V_RMSE_mps']:.6f} / {b['test_WD_RMSE_deg']:.6f}")

    # mission-start
    name="MissionStart_D10_COMMON8"; items={}; phases={}
    for mission in ALL:
        x=build_samples(raws[mission],mission,"mission_start",0,"10min_after_sampling",0,"COMMON8"); items[mission]=x; phases[mission]=x["audit"]["used_phase"]; builds.append({"protocol":name,**x["audit"]})
    rows+=eval_raw_protocol(name,items,{"phase":-1,"anchor":"mission_start","diff":"10min_after_sampling","spline_gap":0,"eligibility":"COMMON8"}); log(f"[MISSION START PHASES] {phases}")

    # differential + eligibility variants
    for name,diff,elig,gap in [
        ("UTCphase0_D1_COMMON8","1min_before_sampling","COMMON8",0),
        ("UTCphase0_D10_BP7","10min_after_sampling","BP7",0),
        ("UTCphase0_D1_BP7","1min_before_sampling","BP7",0),
        ("UTCphase0_D1_BP7_SplineGap1m","1min_before_sampling","BP7",1),
        ("UTCphase0_D1_BP7_SplineGap3m","1min_before_sampling","BP7",3),
        ("UTCphase0_D1_BP7_SplineGap5m","1min_before_sampling","BP7",5),
    ]:
        log(f"[{name}]"); items={}
        for mission in ALL:
            x=build_samples(raws[mission],mission,"UTC",0,diff,gap,elig); items[mission]=x; builds.append({"protocol":name,**x["audit"]})
        rows+=eval_raw_protocol(name,items,{"phase":0,"anchor":"UTC","diff":diff,"spline_gap":gap,"eligibility":elig})

    rdf=pd.DataFrame(rows); pdf=pd.DataFrame(phase_rows); bdf=pd.DataFrame(builds)
    rdf.to_csv(out/"13C_PROTOCOL_RIDGE_AUDIT.csv",index=False,encoding="utf-8-sig")
    pdf.to_csv(out/"13C_PHASE_AUDIT.csv",index=False,encoding="utf-8-sig")
    bdf.to_csv(out/"13C_RAW_PROTOCOL_BUILD_AUDIT.csv",index=False,encoding="utf-8-sig")

    b4=rdf[rdf.model=="Base4-Ridge"].copy(); pb=pdf[pdf.model=="Base4-Ridge"].copy()
    bestv=b4.loc[b4.test_V_RMSE_mps.idxmin()]; bestwd=b4.loc[b4.test_WD_RMSE_deg.idxmin()]
    c4=cur[cur.model=="Base4-Ridge"].iloc[0]
    report={"stage":"13C","script_version":SCRIPT_VERSION,"BP_reference":BP,
            "current12G_Base4":{k:float(c4[k]) for k in ["train_in_sample_U_RMSE_mps","train_in_sample_V_RMSE_mps","train_blocked_OOF_U_RMSE_mps","train_blocked_OOF_V_RMSE_mps","test_U_RMSE_mps","test_V_RMSE_mps","test_WS_RMSE_mps","test_WD_RMSE_deg"]},
            "phase_Base4_V_range":[float(pb.test_V_RMSE_mps.min()),float(pb.test_V_RMSE_mps.max())],
            "phase_Base4_WD_range":[float(pb.test_WD_RMSE_deg.min()),float(pb.test_WD_RMSE_deg.max())],
            "best_posthoc_Base4_V_protocol":bestv.to_dict(),"best_posthoc_Base4_WD_protocol":bestwd.to_dict(),
            "Atlantic_only":atl.to_dict(orient="records"),
            "guardrails":["Post-hoc best phase/protocol is diagnostic only.","Atlantic-only + Tropical validation is not Nie's published split.","If plausible protocol variants do not close V/WD gap, return to two-stage Ridge + residual and optimize repeatable improvement over Ridge."]}
    save_json(out/"13C_REPORT.json",report)

    with open(out/"13C_REPORT.txt","w",encoding="utf-8") as f:
        f.write("13C NIE PROTOCOL FORENSIC RIDGE AUDIT\n"+"="*120+"\n\n")
        f.write("CURRENT EXACT 12G\n"+cur.to_string(index=False)+"\n\nATLANTIC-ONLY\n"+atl.to_string(index=False)+"\n\nPHASE AUDIT\n"+pdf.to_string(index=False)+"\n\nALL PROTOCOLS\n"+rdf.to_string(index=False)+"\n\n")
        f.write(f"Phase Base4 V range: {pb.test_V_RMSE_mps.min():.6f} .. {pb.test_V_RMSE_mps.max():.6f}\n")
        f.write(f"Phase Base4 WD range: {pb.test_WD_RMSE_deg.min():.6f} .. {pb.test_WD_RMSE_deg.max():.6f}\n")
        f.write(f"Best post-hoc V protocol: {bestv.protocol} -> {bestv.test_V_RMSE_mps:.6f}\n")
        f.write(f"Best post-hoc WD protocol: {bestwd.protocol} -> {bestwd.test_WD_RMSE_deg:.6f}\n")

    log("\n"+"="*132); log("13C KEY SUMMARY"); log("="*132)
    log(f"Current Base4 TRAIN in-sample U/V = {c4.train_in_sample_U_RMSE_mps:.6f} / {c4.train_in_sample_V_RMSE_mps:.6f}")
    log(f"Current Base4 TRAIN blocked-OOF U/V = {c4.train_blocked_OOF_U_RMSE_mps:.6f} / {c4.train_blocked_OOF_V_RMSE_mps:.6f}")
    log(f"Current Base4 TROPICAL test U/V = {c4.test_U_RMSE_mps:.6f} / {c4.test_V_RMSE_mps:.6f}")
    log(f"Current Base4 TROPICAL test WS/WD = {c4.test_WS_RMSE_mps:.6f} / {c4.test_WD_RMSE_deg:.6f} deg")
    log(f"UTC phase Base4 V range = {pb.test_V_RMSE_mps.min():.6f} .. {pb.test_V_RMSE_mps.max():.6f}")
    log(f"UTC phase Base4 WD range = {pb.test_WD_RMSE_deg.min():.6f} .. {pb.test_WD_RMSE_deg.max():.6f} deg")
    log(f"Best post-hoc Base4 V protocol = {bestv.protocol} -> {bestv.test_V_RMSE_mps:.6f}")
    log(f"Best post-hoc Base4 WD protocol = {bestwd.protocol} -> {bestwd.test_WD_RMSE_deg:.6f} deg")
    log("\n[IMPORTANT] If 13C cannot materially close V/WD, preserve Ridge and return to two-stage Ridge + residual NN; the objective becomes repeatable improvement over Ridge, not forcing all four metrics below the published BP row.")
    return 0

if __name__=="__main__":
    try: sys.exit(main())
    except Exception:
        print("\n[FATAL ERROR]",flush=True); traceback.print_exc(); sys.exit(1)
