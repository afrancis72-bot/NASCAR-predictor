import re
import unicodedata
import numpy as np
import pandas as pd

NASCAR_STANDINGS_URL="https://www.nascar.com/standings/nascar-cup-series/"
NASCAR_DRIVER_URL="https://www.nascar.com/drivers/{slug}/"

def _slug(name):
    s=unicodedata.normalize("NFKD",str(name)).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+","-",s).strip("-")

def _flatten(df):
    d=df.copy()
    if isinstance(d.columns,pd.MultiIndex):
        d.columns=[" ".join(str(x) for x in c if str(x)!="nan").strip() for c in d.columns]
    else:
        d.columns=[str(c).strip() for c in d.columns]
    return d

def _col(cols, keys):
    for c in cols:
        low=c.lower()
        if any(k in low for k in keys): return c
    return None

def fetch_official_season_stats():
    try:
        tables=pd.read_html(NASCAR_STANDINGS_URL)
    except Exception as e:
        return pd.DataFrame(), f"Official NASCAR standings unavailable: {e}"
    best=None
    for t in tables:
        t=_flatten(t)
        lc=[c.lower() for c in t.columns]
        if any("driver" in c for c in lc) and (any("dnf" in c for c in lc) or any("laps led" in c for c in lc)):
            if best is None or len(t)>len(best): best=t
    if best is None:
        return pd.DataFrame(),"Official NASCAR page loaded, but its season-stat table was not exposed to the app."
    nc=_col(best.columns,["driver"])
    out=pd.DataFrame({"Name":best[nc].astype(str).str.replace(r"\s+"," ",regex=True).str.strip()})
    mapping={"season_wins":["wins","win"],"season_top5":["top 5","top5"],
             "season_top10":["top 10","top10"],"season_dnfs":["dnf"],
             "season_laps_led":["laps led"],"season_starts":["starts"]}
    for new,keys in mapping.items():
        c=_col(best.columns,keys)
        out[new]=pd.to_numeric(best[c],errors="coerce") if c else np.nan
    return out.drop_duplicates("Name"),"Official NASCAR season stats loaded."

def fetch_driver_race_history(name):
    try:
        tables=pd.read_html(NASCAR_DRIVER_URL.format(slug=_slug(name)))
    except Exception:
        return pd.DataFrame()
    good=[]
    for t in tables:
        t=_flatten(t)
        lc=[c.lower() for c in t.columns]
        if any("track" in c for c in lc) and any("finish" in c for c in lc): good.append(t)
    return max(good,key=len) if good else pd.DataFrame()

def _history_features(h, track):
    if h.empty:return {}
    tc=_col(h.columns,["track"]); fc=_col(h.columns,["finish"])
    sc=_col(h.columns,["start"]); lc=_col(h.columns,["laps led"])
    if not tc or not fc:return {}
    x=h.copy()
    x["_f"]=pd.to_numeric(x[fc],errors="coerce")
    x["_s"]=pd.to_numeric(x[sc],errors="coerce") if sc else np.nan
    x["_l"]=pd.to_numeric(x[lc],errors="coerce") if lc else np.nan
    x=x[x["_f"].notna()]
    recent=x.head(6)
    key=str(track).split(" Motor")[0].split(" Raceway")[0]
    exact=x[x[tc].astype(str).str.contains(key,case=False,na=False)]
    return {"recent_avg_finish":recent["_f"].mean(),
            "recent_avg_start":recent["_s"].mean(),
            "recent_laps_led":recent["_l"].sum(min_count=1),
            "track_history_avg_finish":exact["_f"].mean() if len(exact) else np.nan,
            "track_history_laps_led":exact["_l"].sum(min_count=1) if len(exact) else np.nan,
            "history_races_found":len(x),"track_races_found":len(exact)}

def build_auto_intelligence(dk, selected_track, max_driver_pages=50):
    season,status=fetch_official_season_stats()
    base=pd.DataFrame({"Name":dk["Name"].astype(str)})
    if not season.empty: base=base.merge(season,on="Name",how="left")
    else:
        for c in ["season_wins","season_top5","season_top10","season_dnfs","season_laps_led","season_starts"]:
            base[c]=np.nan
    starts=pd.to_numeric(base["season_starts"],errors="coerce").replace(0,np.nan)
    base["season_dnf_rate"]=pd.to_numeric(base["season_dnfs"],errors="coerce")/starts
    rows=[]
    for name in base["Name"].head(max_driver_pages):
        row={"Name":name}
        row.update(_history_features(fetch_driver_race_history(name),selected_track))
        rows.append(row)
    if rows: base=base.merge(pd.DataFrame(rows),on="Name",how="left")
    base["auto_intel_source"]="NASCAR.com official"
    return base,status
