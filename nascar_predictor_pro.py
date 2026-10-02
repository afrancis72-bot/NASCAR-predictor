
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

FINISH_POINTS = {
1:45,2:42,3:41,4:40,5:39,6:38,7:37,8:36,9:35,10:34,
11:32,12:31,13:30,14:29,15:28,16:27,17:26,18:25,19:24,20:23,
21:21,22:20,23:19,24:18,25:17,26:16,27:15,28:14,29:13,30:12,
31:10,32:9,33:8,34:7,35:6,36:5,37:4,38:3,39:2,40:1
}

@dataclass
class Config:
    sims: int = 25000
    seed: int = 42
    salary_cap: int = 50000
    roster_size: int = 6
    laps: int = 267
    max_overlap: int = 4

def _z(s, higher_better=True):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=s.index)
    fill = s.median()
    x = s.fillna(fill)
    sd = x.std(ddof=0)
    z = (x-x.mean())/(sd if sd > 1e-9 else 1.0)
    return z if higher_better else -z

def _rank_signal(df):
    cols = ["avg20_rank","avg15_rank","avg10_rank","avg5_rank","single_lap_rank"]
    weights = [0.30,0.25,0.22,0.15,0.08]
    vals = []
    wts = []
    for c,w in zip(cols,weights):
        if c in df:
            vals.append(_z(df[c], higher_better=False)*w)
            wts.append(df[c].notna().astype(float)*w)
    if not vals:
        return pd.Series(0.0,index=df.index)
    num = sum(vals)
    den = sum(wts).replace(0,np.nan)
    return (num/den).fillna(0)

def build_features(dk, priors, track, updates=None):
    df = dk.copy()
    df = df.merge(priors.drop(columns=["Salary","AvgPointsPerGame"],errors="ignore"), on="Name", how="left")
    if updates is not None:
        df = df.merge(updates, on="Name", how="left")
    else:
        for c in ["single_lap_rank","avg5_rank","avg10_rank","avg15_rank","avg20_rank","qualifying_position"]:
            df[c] = np.nan

    # Baselines. Missing specialized stats are shrunk to field-neutral, never treated as zero/bad.
    df["dk_base_z"] = _z(df["AvgPointsPerGame"], True)
    df["intermediate_z"] = _z(df["intermediate_2026_avg_finish"], False)
    df["vegas_spring_z"] = _z(df["vegas_spring_2026_finish"], False)
    df["vegas_hist_z"] = _z(df["vegas_nextgen_avg_finish"], False)
    df["season_finish_z"] = _z(df["season_2026_avg_finish"], False)
    df["season_led_z"] = _z(df["season_2026_laps_led"], True)
    df["intermediate_led_z"] = _z(df["intermediate_2026_laps_led"], True)
    df["vegas_led_z"] = _z(df.get("vegas_nextgen_laps_led", pd.Series(np.nan,index=df.index)), True)
    df["manual_speed_z"] = _z(df["manual_speed_rating"], True)
    df["practice_z"] = _rank_signal(df)

    # Blend venue history only where evidence exists; neutral shrinkage elsewhere.
    # Reliability shrinkage: specialized fields are powerful but incomplete in V1A.
    # DK baseline is the broad field prior; sparse official stats adjust it rather than replace it.
    vegas = 0.45*df["vegas_spring_z"].clip(-2,2) + 0.20*df["vegas_hist_z"].clip(-2,2) + 0.35*df["vegas_led_z"].clip(-2,2)
    current = 0.82*df["dk_base_z"] + 0.10*df["season_finish_z"].clip(-2,2) + 0.08*df["manual_speed_z"].clip(-2,2)
    intermediate = 0.78*df["intermediate_z"].clip(-2,2) + 0.22*df["intermediate_led_z"].clip(-2,2)

    # Pre-qualifying baseline; practice signal only becomes material when populated.
    practice_available = df[["single_lap_rank","avg5_rank","avg10_rank","avg15_rank","avg20_rank"]].notna().any(axis=1)
    qual_available = df["qualifying_position"].notna()
    qual_z = _z(df["qualifying_position"], False) if qual_available.any() else pd.Series(0.0,index=df.index)

    w = track.iloc[0]
    # V1A calibrated blend: broad prior carries extra weight until full historical ingestion is built.
    raw = (0.34*current + 0.23*intermediate + 0.15*vegas + 0.28*df["dk_base_z"])
    # Reallocate unavailable live-data weights to the stable baseline rather than pretending zero.
    if practice_available.any():
        raw += w.weight_practice*df["practice_z"]
    else:
        raw += 0.04*(0.50*intermediate + 0.50*current)
    if qual_available.any():
        raw += w.weight_qualifying*qual_z
    else:
        raw += 0.03*(0.50*intermediate + 0.50*vegas)

    df["race_strength"] = raw
    df["projected_start"] = df["qualifying_position"].fillna((len(df)+1)/2)
    # Dominator score deliberately differs from finish strength.
    start_front = _z(df["projected_start"], False)
    df["dominator_strength"] = (
        w.dominator_weight_speed*(0.65*current+0.35*df["practice_z"]) +
        w.dominator_weight_intermediate*(0.55*intermediate+0.45*df["intermediate_led_z"]) +
        w.dominator_weight_vegas*(0.35*vegas+0.35*df["season_led_z"]+0.30*df["vegas_led_z"]) +
        w.dominator_weight_start*start_front
    )
    df["data_coverage"] = (
        df[["intermediate_2026_avg_finish","vegas_spring_2026_finish",
            "season_2026_avg_finish","season_2026_laps_led"]].notna().sum(axis=1)
        + practice_available.astype(int)*2 + qual_available.astype(int)
    )
    return df

def simulate(features, config=Config()):
    rng = np.random.default_rng(config.seed)
    n = len(features); sims = config.sims
    strength = features["race_strength"].to_numpy(float)
    dom = features["dominator_strength"].to_numpy(float)
    starts = features["projected_start"].to_numpy(float)
    dnf = pd.to_numeric(features.get("manual_dnf_risk", pd.Series(np.nan,index=features.index)), errors="coerce").fillna(0.11).clip(0.02,0.35).to_numpy()

    finish = np.empty((sims,n), dtype=np.int16)
    laps_led = np.zeros((sims,n), dtype=np.int16)
    fastest = np.zeros((sims,n), dtype=np.int16)
    dkpts = np.zeros((sims,n), dtype=np.float32)

    # Simulation: latent performance + incident tail. V1 intentionally transparent.
    for s in range(sims):
        latent = strength + rng.normal(0, 1.55, n)
        incidents = rng.random(n) < dnf
        latent[incidents] -= rng.uniform(2.5,5.5,incidents.sum())
        order = np.argsort(-latent)
        f = np.empty(n,dtype=np.int16); f[order]=np.arange(1,n+1)
        finish[s]=f

        # Allocate dominator events using Dirichlet probabilities tied to latent speed.
        dscore = 0.52*dom + 0.48*latent
        p = np.exp(dscore - dscore.max()); p /= p.sum()
        ll = rng.multinomial(config.laps, p)
        flp = rng.multinomial(config.laps, np.sqrt(p)/np.sqrt(p).sum())
        laps_led[s]=ll; fastest[s]=flp

        fp = np.array([FINISH_POINTS.get(int(x),1) for x in f])
        pdiff = starts - f
        dkpts[s] = fp + pdiff + 0.25*ll + 0.45*flp

    out = features[["Name","ID","Salary","AvgPointsPerGame","race_strength","dominator_strength","data_coverage","projected_start"]].copy()
    out["proj_dk"] = dkpts.mean(axis=0)
    out["floor_p20"] = np.quantile(dkpts,0.20,axis=0)
    out["ceiling_p90"] = np.quantile(dkpts,0.90,axis=0)
    out["win_pct"] = (finish==1).mean(axis=0)
    out["top5_pct"] = (finish<=5).mean(axis=0)
    out["top10_pct"] = (finish<=10).mean(axis=0)
    out["exp_finish"] = finish.mean(axis=0)
    out["exp_laps_led"] = laps_led.mean(axis=0)
    out["exp_fastest_laps"] = fastest.mean(axis=0)
    out["value_per_1k"] = out["proj_dk"]/(out["Salary"]/1000)
    return out.sort_values("proj_dk",ascending=False).reset_index(drop=True), dkpts

def optimize_lineups(proj, n_lineups=10, salary_cap=50000, roster_size=6, max_overlap=4, objective="ceiling"):
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except Exception as e:
        raise RuntimeError("SciPy with scipy.optimize.milp is required for lineup optimization.") from e
    score_col = {"median":"proj_dk","ceiling":"ceiling_p90","value":"value_per_1k"}.get(objective,"ceiling_p90")
    scores = proj[score_col].to_numpy(float)
    sal = proj["Salary"].to_numpy(float)
    n=len(proj); lineups=[]
    constraints=[LinearConstraint(np.ones((1,n)),[roster_size],[roster_size]),
                 LinearConstraint(sal.reshape(1,-1),[-np.inf],[salary_cap])]
    for _ in range(n_lineups):
        cons=list(constraints)
        for prev in lineups:
            a=np.zeros(n); a[prev]=1
            cons.append(LinearConstraint(a.reshape(1,-1),[-np.inf],[max_overlap]))
        res=milp(c=-scores, integrality=np.ones(n), bounds=Bounds(np.zeros(n),np.ones(n)), constraints=cons)
        if not res.success: break
        idx=np.where(res.x>0.5)[0].tolist()
        lineups.append(idx)
    rows=[]
    for k,idx in enumerate(lineups,1):
        sub=proj.iloc[idx]
        rows.append({
            "Lineup":k,
            "Drivers":" | ".join(sub["Name"]),
            "Salary":int(sub["Salary"].sum()),
            "Projected":round(sub["proj_dk"].sum(),2),
            "CeilingScore":round(sub["ceiling_p90"].sum(),2),
            "IDs":" | ".join(sub["ID"].astype(str))
        })
    return pd.DataFrame(rows)
