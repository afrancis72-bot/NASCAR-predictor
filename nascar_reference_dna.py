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

def _flat(t):
    x=t.copy()
    if isinstance(x.columns,pd.MultiIndex):
        x.columns=[" ".join(str(v) for v in c if str(v)!="nan").strip() for c in x.columns]
    else:x.columns=[str(c).strip() for c in x.columns]
    return x

def _find(cols,keys):
    for c in cols:
        z=c.lower()
        if any(k in z for k in keys):return c
    return None

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
    tabs=[_flat(t) for t in _tables(html)]
    for t in tabs:
        fc=_find(t.columns,["finish"]); sc=_find(t.columns,["start"])
        rc=_find(t.columns,["race"]); ser=_find(t.columns,["series"])
        if fc and sc and rc:
            x=t.copy()
            if ser:x=x[x[ser].astype(str).str.contains("Cup",case=False,na=False)]
            x["FinishNum"]=pd.to_numeric(x[fc],errors="coerce")
            x["StartNum"]=pd.to_numeric(x[sc],errors="coerce")
            x=x[x["FinishNum"].notna()]
            if len(x):return x.head(8),rc
    return pd.DataFrame(),None

def track_cup_rows(html,track_key):
    tabs=[_flat(t) for t in _tables(html)]
    for t in tabs:
        tc=_find(t.columns,["track"]); fc=_find(t.columns,["finish"])
        ser=_find(t.columns,["series"]); lc=_find(t.columns,["laps led"])
        if tc and fc:
            x=t.copy()
            if ser:x=x[x[ser].astype(str).str.contains("Cup",case=False,na=False)]
            x=x[x[tc].astype(str).str.contains(track_key,case=False,na=False)]
            if len(x):
                x["FinishNum"]=pd.to_numeric(x[fc],errors="coerce")
                x["LapsLedNum"]=pd.to_numeric(x[lc],errors="coerce") if lc else np.nan
                return x
    return pd.DataFrame()

def build_reference_dna(dk, selected_track):
    # Atlanta was renamed EchoPark Speedway in the current source; modern Atlanta is a drafting track.
    key="EchoPark" if "Atlanta" in selected_track else selected_track.split(" Motor")[0].split(" Speedway")[0].split(" Raceway")[0]
    rows=[]
    for name in dk["Name"].astype(str):
        row={"Name":name}
        try:
            html=driver_page(name)
            recent,_=recent_cup_rows(html)
            trk=track_cup_rows(html,key)
            row["recent_races"]=len(recent)
            row["recent_avg_finish"]=recent["FinishNum"].mean() if len(recent) else np.nan
            row["recent_avg_start"]=recent["StartNum"].mean() if len(recent) else np.nan
            row["recent_place_diff"]=(recent["StartNum"]-recent["FinishNum"]).mean() if len(recent) else np.nan
            row["track_races"]=len(trk)
            row["track_history_avg_finish"]=trk["FinishNum"].mean() if len(trk) else np.nan
            row["track_history_laps_led"]=trk["LapsLedNum"].mean() if len(trk) else np.nan
            row["reference_ok"]=True
        except Exception:
            row.update({"recent_races":0,"recent_avg_finish":np.nan,"recent_avg_start":np.nan,
                        "recent_place_diff":np.nan,"track_races":0,"track_history_avg_finish":np.nan,
                        "track_history_laps_led":np.nan,"reference_ok":False})
        rows.append(row)
    x=pd.DataFrame(rows)
    # Component scores are relative to the current DK field, so 50 ~= field median.
    x["recent_form_score"]=_pct_score(x["recent_avg_finish"],invert=True)
    x["place_diff_score"]=_pct_score(x["recent_place_diff"])
    x["track_fit_score"]=_pct_score(x["track_history_avg_finish"],invert=True)
    x["dominator_score"]=_pct_score(x["track_history_laps_led"])
    # Reliability proxy from observed recent finishes: reward sample coverage and stronger finish profile.
    x["reliability_score"]=x["recent_form_score"]
    comps=["recent_form_score","place_diff_score","track_fit_score","dominator_score","reliability_score"]
    x["reference_components_available"]=x[comps].notna().sum(axis=1)
    x["reference_coverage"]=x["reference_components_available"]/len(comps)
    return x

def blend_driver_dna(dk, ref):
    z=dk[["Name","Salary","AvgPointsPerGame"]].merge(ref,on="Name",how="left")
    z["dk_market_score"]=(0.55*_pct_score(z["AvgPointsPerGame"])+0.45*_pct_score(z["Salary"])).fillna(50)
    weights={"recent_form_score":.25,"place_diff_score":.10,"track_fit_score":.25,
             "dominator_score":.15,"reliability_score":.10,"dk_market_score":.15}
    vals=[]
    for _,r in z.iterrows():
        num=den=0.0
        for c,w in weights.items():
            v=r.get(c,np.nan)
            if pd.notna(v):num+=float(v)*w;den+=w
        vals.append(num/den if den else 50.0)
    z["driver_dna_score"]=vals
    z["data_coverage_pct"]=(15+85*z["reference_coverage"].fillna(0)).round(0)
    z["model_confidence"]=np.where(z["data_coverage_pct"]>=80,"HIGH",
                           np.where(z["data_coverage_pct"]>=55,"MEDIUM","LOW"))
    return z
