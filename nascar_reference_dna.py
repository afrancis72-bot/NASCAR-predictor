import re, unicodedata
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import numpy as np, pandas as pd

BASE="https://nascar-reference.com"
UA="Mozilla/5.0 (compatible; NASCAR-Predictor/2.5)"

def _get(path,timeout=12):
    req=Request(BASE+path,headers={"User-Agent":UA,"Accept":"text/html,*/*"})
    with urlopen(req,timeout=timeout) as r:
        return r.read().decode("utf-8","replace")

def _slug(name):
    s=unicodedata.normalize("NFKD",str(name)).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+","-",s).strip("-")

def _tables(html):
    try:return pd.read_html(html)
    except:return []

def _normcol(c):
    if isinstance(c,tuple): c=" ".join(str(v) for v in c if str(v).lower()!="nan")
    c=unicodedata.normalize("NFKD",str(c)).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+"," ",c).strip()

def _flat(t):
    x=t.copy(); x.columns=[_normcol(c) for c in x.columns]; return x

def _find(cols,keys):
    keys=[_normcol(k) for k in keys]
    for k in keys:
        for c in cols:
            if c==k:return c
    for k in keys:
        for c in cols:
            if k in c:return c
    return None

def _num(v):
    return pd.to_numeric(pd.Series(v).astype(str).str.extract(r"([-+]?\d+(?:\.\d+)?)",expand=False),errors="coerce")

def _z(series, invert=False):
    x=pd.to_numeric(series,errors="coerce")
    if x.notna().sum()<2:return pd.Series(np.nan,index=x.index)
    sd=x.std(ddof=0)
    z=(x-x.mean())/(sd if sd and np.isfinite(sd) else 1)
    return -z if invert else z

def _pct_score(series,invert=False):
    x=pd.to_numeric(series,errors="coerce")
    if x.notna().sum()<2:return pd.Series(np.nan,index=x.index)
    r=x.rank(pct=True)
    if invert:r=1-r+1/len(x.dropna())
    return (r*100).clip(0,100)

def driver_page(name):
    return _get("/drivers/"+_slug(name))

def recent_cup_rows(html):
    candidates=[]
    for t in [_flat(t) for t in _tables(html)]:
        fc=_find(t.columns,["finish"]); sc=_find(t.columns,["start"]); rc=_find(t.columns,["race"]); ser=_find(t.columns,["series"])
        if fc and sc and rc:
            x=t.copy()
            if ser: x=x[x[ser].astype(str).str.strip().str.lower().eq("cup")]
            x["FinishNum"]=_num(x[fc]); x["StartNum"]=_num(x[sc])
            x=x[x["FinishNum"].notna() & x["StartNum"].notna()]
            if len(x): candidates.append(x)
    return (max(candidates,key=len).head(20), "race") if candidates else (pd.DataFrame(),None)

def best_track_rows(html,track_key):
    for t in [_flat(t) for t in _tables(html)]:
        tc=_find(t.columns,["track"]); rc=_find(t.columns,["races"]); af=_find(t.columns,["avg finish"])
        if tc and rc and af:
            hit=t[t[tc].astype(str).str.contains(track_key,case=False,na=False,regex=False)].copy()
            if len(hit):
                hit["TrackRaces"]=_num(hit[rc]); hit["TrackAvgFinish"]=_num(hit[af]); return hit
    return pd.DataFrame()

def profile_metric(html,label):
    txt=" ".join(BeautifulSoup(html,"html.parser").stripped_strings)
    m=re.search(rf"([-+]?\d+(?:\.\d+)?)\s+{re.escape(label)}",txt,re.I)
    return float(m.group(1)) if m else np.nan

def build_reference_dna(dk, selected_track):
    key="EchoPark" if "Atlanta" in selected_track else selected_track.split(" Motor")[0].split(" Speedway")[0].split(" Raceway")[0]
    atl="Atlanta" in selected_track
    rows=[]
    for name in dk["Name"].astype(str):
        row={"Name":name}
        try:
            html=driver_page(name); recent,_=recent_cup_rows(html); trk=best_track_rows(html,key)
            row["recent_races"]=len(recent)
            row["recent_avg_finish"]=recent["FinishNum"].head(10).mean() if len(recent) else np.nan
            row["recent_avg_start"]=recent["StartNum"].head(10).mean() if len(recent) else np.nan
            row["recent_place_diff"]=(recent["StartNum"].head(10)-recent["FinishNum"].head(10)).mean() if len(recent) else np.nan
            row["track_races"]=float(trk["TrackRaces"].iloc[0]) if len(trk) else np.nan
            row["track_history_avg_finish"]=float(trk["TrackAvgFinish"].iloc[0]) if len(trk) else np.nan
            row["comparable_track_avg_finish"]=profile_metric(html,"Superspeedway") if atl else np.nan
            row["source_reliability"]=profile_metric(html,"Reliability")
            row["reference_ok"]=True
        except Exception:
            row.update({"recent_races":0,"recent_avg_finish":np.nan,"recent_avg_start":np.nan,"recent_place_diff":np.nan,
                        "track_races":np.nan,"track_history_avg_finish":np.nan,"comparable_track_avg_finish":np.nan,
                        "source_reliability":np.nan,"reference_ok":False})
        rows.append(row)
    x=pd.DataFrame(rows)
    x["recent_form_score"]=_pct_score(x["recent_avg_finish"],invert=True)
    x["place_diff_score"]=_pct_score(x["recent_place_diff"])
    x["track_fit_score"]=_pct_score(x["track_history_avg_finish"],invert=True)
    x["comparable_track_score"]=_pct_score(x["comparable_track_avg_finish"],invert=True)
    x["reliability_score"]=pd.to_numeric(x["source_reliability"],errors="coerce").clip(0,100)
    comps=["recent_form_score","place_diff_score","track_fit_score","comparable_track_score","reliability_score"]
    x["reference_components_available"]=x[comps].notna().sum(axis=1)
    x["reference_coverage"]=x["reference_components_available"]/len(comps)
    return x

def blend_driver_dna(dk, ref):
    z=dk[["Name","Salary","AvgPointsPerGame"]].merge(ref,on="Name",how="left")
    z["dk_market_score"]=(0.55*_pct_score(z["AvgPointsPerGame"])+0.45*_pct_score(z["Salary"])).fillna(50)
    weights={"recent_form_score":.25,"place_diff_score":.10,"track_fit_score":.20,
             "comparable_track_score":.20,"reliability_score":.10,"dk_market_score":.15}
    vals=[]
    for _,r in z.iterrows():
        num=den=0.0
        for c,w in weights.items():
            v=r.get(c,np.nan)
            if pd.notna(v): num+=float(v)*w; den+=w
        vals.append(num/den if den else 50.0)
    z["driver_dna_score"]=vals
    z["data_coverage_pct"]=(15+85*z["reference_coverage"].fillna(0)).round(0)
    z["model_confidence"]=np.where(z["data_coverage_pct"]>=80,"HIGH",np.where(z["data_coverage_pct"]>=55,"MEDIUM","LOW"))
    return z
