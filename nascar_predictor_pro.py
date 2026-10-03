
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


def _safe_signal(s, higher_is_better=True, min_obs=6, cap=2.25):
    x=pd.to_numeric(s,errors="coerce")
    obs=x.notna()
    n=int(obs.sum())
    out=pd.Series(0.0,index=x.index,dtype=float)
    if n < min_obs:
        return out
    vals=x[obs]
    sd=float(vals.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-9:
        return out
    z=(vals-float(vals.mean()))/sd
    if not higher_is_better:
        z=-z
    reliability=min(1.0, n/max(min_obs*2,12))
    out.loc[obs]=z.clip(-cap,cap)*reliability
    return out

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
    df["intermediate_z"] = _safe_signal(df.get("intermediate_2026_avg_finish", pd.Series(np.nan,index=df.index)), False, min_obs=6)
    df["vegas_spring_z"] = _safe_signal(df.get("vegas_spring_2026_finish", pd.Series(np.nan,index=df.index)), False, min_obs=6)
    df["vegas_hist_z"] = _safe_signal(df.get("vegas_nextgen_avg_finish", pd.Series(np.nan,index=df.index)), False, min_obs=6)
    df["season_finish_z"] = _safe_signal(df.get("season_2026_avg_finish", pd.Series(np.nan,index=df.index)), False, min_obs=6)
    df["season_led_z"] = _safe_signal(df.get("season_2026_laps_led", pd.Series(np.nan,index=df.index)), True, min_obs=6)
    df["intermediate_led_z"] = _safe_signal(df.get("intermediate_2026_laps_led", pd.Series(np.nan,index=df.index)), True, min_obs=6)
    df["vegas_led_z"] = _safe_signal(df.get("vegas_nextgen_laps_led", pd.Series(np.nan,index=df.index)), True, min_obs=6)
    df["vegas_spring_led_z"] = _safe_signal(df.get("vegas_spring_2026_laps_led", pd.Series(np.nan,index=df.index)), True, min_obs=6)
    df["nextgen_1p5_led_z"] = _safe_signal(df.get("nextgen_1p5_laps_led", pd.Series(np.nan,index=df.index)), True, min_obs=6)
    df["season_speed_z"] = _safe_signal(df.get("season_speed_rank", pd.Series(np.nan,index=df.index)), False, min_obs=6)
    df["chase_rating_z"] = _safe_signal(df.get("chase_driver_rating", pd.Series(np.nan,index=df.index)), True, min_obs=6)
    df["chase_finish_z"] = _safe_signal(df.get("chase_avg_finish", pd.Series(np.nan,index=df.index)), False, min_obs=6)
    df["vegas_career_finish_z"] = _safe_signal(df.get("vegas_career_avg_finish", pd.Series(np.nan,index=df.index)), False, min_obs=6)
    df["vegas_career_led_z"] = _safe_signal(df.get("vegas_career_laps_led", pd.Series(np.nan,index=df.index)), True, min_obs=6)
    df["manual_speed_z"] = _z(df["manual_speed_rating"], True)
    df["practice_z"] = _rank_signal(df)

    # Blend venue history only where evidence exists; neutral shrinkage elsewhere.
    # V1.2 reliability-aware layers. Sparse evidence adjusts the broad DK prior,
    # but no driver is rewarded simply because more columns happen to be populated.
    vegas = (0.45*df["vegas_spring_z"].clip(-2,2) +
             0.15*df["vegas_hist_z"].clip(-2,2) +
             0.15*df["vegas_led_z"].clip(-2,2) +
             0.15*df["vegas_career_finish_z"].clip(-2,2) +
             0.10*df["vegas_career_led_z"].clip(-2,2))
    current = (0.30*df["dk_base_z"].clip(-2,2) +
               0.30*df["season_finish_z"].clip(-2,2) +
               0.15*df["season_speed_z"].clip(-2,2) +
               0.15*df["chase_rating_z"].clip(-2,2) +
               0.10*df["chase_finish_z"].clip(-2,2))
    intermediate = (0.55*df["intermediate_z"].clip(-2,2) +
                    0.15*df["intermediate_led_z"].clip(-2,2) +
                    0.30*df["nextgen_1p5_led_z"].clip(-2,2))

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
        w.dominator_weight_intermediate*(0.40*intermediate+0.25*df["intermediate_led_z"]+0.35*df["nextgen_1p5_led_z"]) +
        w.dominator_weight_vegas*(0.25*vegas+0.25*df["season_led_z"]+0.25*df["vegas_led_z"]+0.25*df["vegas_spring_led_z"]) +
        w.dominator_weight_start*start_front
    )
    df["place_diff_room"] = df["projected_start"] - 1
    df["front_start_score"] = ((10 - df["projected_start"]).clip(lower=0) / 9.0)
    df["back_start_score"] = ((df["projected_start"] - 20).clip(lower=0) / max(len(df)-20,1))
    # Starting up front modestly increases early dominator access; it does NOT redefine driver ability.
    if qual_available.any():
        df["dominator_strength"] += 0.22*_z(df["projected_start"], False)
    evidence_cols=["intermediate_2026_avg_finish","vegas_spring_2026_finish",
        "season_2026_avg_finish","season_2026_laps_led","nextgen_1p5_laps_led",
        "season_speed_rank","chase_driver_rating","vegas_career_avg_finish"]
    df["data_coverage"] = df[evidence_cols].notna().sum(axis=1) + practice_available.astype(int)*2 + qual_available.astype(int)
    df["model_confidence"] = (0.45 + 0.055*df["data_coverage"]).clip(upper=0.95)
    df["audit_dk_prior"] = df["dk_base_z"].clip(-2,2)
    df["audit_vegas"] = vegas
    df["audit_current"] = current
    df["audit_intermediate"] = intermediate
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
        sigma = 1.85 - 0.65*features["model_confidence"].to_numpy(float)
        # V1.4.1: modest track-position anchor. Starting position is predictive of finish,
        # so front starters retain some track-position value and deep starters must earn PD.
        # This changes finish probabilities, not DraftKings scoring.
        start_anchor = -((starts - starts.mean()) / (starts.std() if starts.std() > 1e-9 else 1.0))
        latent = strength + 0.22*start_anchor + rng.normal(0, sigma, n)
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

    out = features[["Name","ID","Salary","AvgPointsPerGame","race_strength","dominator_strength",
                    "data_coverage","model_confidence","projected_start","place_diff_room",
                    "front_start_score","back_start_score","audit_dk_prior","audit_vegas","audit_current","audit_intermediate"]].copy()
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


def optimize_race_sim_lineups(proj, sim_matrix, n_lineups=10, salary_cap=50000,
                              roster_size=6, max_overlap=4, candidate_pool=2500,
                              seed=42, risk_mode="gpp"):
    """
    V1.1 optimizer. Generates legal candidate rosters, then evaluates each roster by
    summing its six drivers inside EACH SAME simulated race. This preserves NASCAR
    race-level correlation and makes P90/P95 true lineup outcomes rather than sums
    of individual percentiles.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    rng=np.random.default_rng(seed)
    n=len(proj)
    salary=proj["Salary"].to_numpy(float)
    # Candidate generation: perturb mean projection to create diverse, strong legal rosters.
    base=proj["proj_dk"].to_numpy(float)
    candidates=set()
    attempts=max(candidate_pool*3,3000)
    for _ in range(attempts):
        noise=rng.normal(0,7.5,n)
        score=base+noise
        cons=[
            LinearConstraint(np.ones((1,n)),[roster_size],[roster_size]),
            LinearConstraint(salary.reshape(1,-1),[-np.inf],[salary_cap])
        ]
        res=milp(c=-score,integrality=np.ones(n),
                 bounds=Bounds(np.zeros(n),np.ones(n)),constraints=cons)
        if res.success:
            idx=tuple(np.where(res.x>0.5)[0].tolist())
            candidates.add(idx)
        if len(candidates)>=candidate_pool: break

    # Always include mean-optimal lineup.
    cons=[LinearConstraint(np.ones((1,n)),[roster_size],[roster_size]),
          LinearConstraint(salary.reshape(1,-1),[-np.inf],[salary_cap])]
    res=milp(c=-base,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),constraints=cons)
    if res.success:candidates.add(tuple(np.where(res.x>0.5)[0].tolist()))

    rows=[]
    for idx in candidates:
        scores=sim_matrix[:,idx].sum(axis=1)
        mean=float(scores.mean()); p10=float(np.quantile(scores,.10))
        p90=float(np.quantile(scores,.90)); p95=float(np.quantile(scores,.95))
        # GPP objective rewards coherent upper-tail outcomes without ignoring mean.
        gpp=0.40*mean+0.25*p90+0.35*p95
        cash=0.65*mean+0.35*p10
        rows.append((idx,mean,p10,p90,p95,gpp,cash))
    rows.sort(key=lambda x:x[5] if risk_mode=="gpp" else x[6],reverse=True)

    selected=[]
    for r in rows:
        idx=r[0]
        if all(len(set(idx)&set(s[0]))<=max_overlap for s in selected):
            selected.append(r)
            if len(selected)>=n_lineups: break

    out=[]
    for k,(idx,mean,p10,p90,p95,gpp,cash) in enumerate(selected,1):
        sub=proj.iloc[list(idx)]
        out.append({
            "Lineup":k,
            "Drivers":" | ".join(sub["Name"]),
            "Salary":int(sub["Salary"].sum()),
            "Mean":round(mean,2),"P10":round(p10,2),
            "P90":round(p90,2),"P95":round(p95,2),
            "GPP_Score":round(gpp,2),
            "IDs":" | ".join(sub["ID"].astype(str))
        })
    return pd.DataFrame(out)


def portfolio_exposure(lineups, proj):
    if lineups is None or len(lineups)==0:
        return pd.DataFrame()
    counts={}
    for s in lineups["Drivers"]:
        for name in [x.strip() for x in str(s).split("|")]:
            counts[name]=counts.get(name,0)+1
    n=len(lineups)
    out=pd.DataFrame({"Name":list(counts.keys()),"Lineups":list(counts.values())})
    out["Exposure"]=out["Lineups"]/n
    return out.merge(proj[["Name","Salary","proj_dk","ceiling_p90","model_confidence"]],
                     on="Name",how="left").sort_values(["Exposure","proj_dk"],ascending=[False,False])


def classify_race_scripts(features, sim_matrix):
    """Classify each coherent simulated race into a DFS-relevant race script."""
    names=features["Name"].tolist()
    dom=features["dominator_strength"].to_numpy(float)
    elite_idx=np.argsort(dom)[::-1][:4]
    ham_idx=names.index("Denny Hamlin") if "Denny Hamlin" in names else None
    lar_idx=names.index("Kyle Larson") if "Kyle Larson" in names else None

    # A high-scoring driver is a practical proxy for dominator/PD concentration
    # because sim_matrix already includes finishing, PD, laps led and fastest laps.
    order=np.argsort(sim_matrix,axis=1)[:,::-1]
    top1=sim_matrix[np.arange(len(sim_matrix)),order[:,0]]
    top2=sim_matrix[np.arange(len(sim_matrix)),order[:,1]]
    top3=sim_matrix[np.arange(len(sim_matrix)),order[:,2]]
    spread=top1-top3

    labels=np.full(len(sim_matrix),"split_dominator",dtype=object)
    # Chaos: unusually compressed top scores / multiple value paths.
    labels[spread < np.quantile(spread,.22)]="chaos_attrition"
    # Track-position / concentrated domination.
    labels[spread > np.quantile(spread,.78)]="track_position"

    if ham_idx is not None:
        ham=sim_matrix[:,ham_idx]
        labels[(ham >= np.quantile(ham,.82)) & (ham >= top2)]="hamlin_dominant"
    if lar_idx is not None:
        lar=sim_matrix[:,lar_idx]
        labels[(lar >= np.quantile(lar,.82)) & (lar >= top2)]="larson_dominant"
    return labels

def _candidate_lineup_matrix(lineups, features):
    name_to_idx={n:i for i,n in enumerate(features["Name"])}
    mats=[]
    valid=[]
    for ridx,row in lineups.iterrows():
        ns=[x.strip() for x in str(row["Drivers"]).split("|")]
        if len(ns)==6 and all(n in name_to_idx for n in ns):
            mats.append([name_to_idx[n] for n in ns])
            valid.append(ridx)
    return np.asarray(mats,dtype=int), valid

def optimize_scenario_portfolio(features, sim_matrix, n_lineups=10, salary_cap=50000,
                                roster_size=6, max_overlap=4, candidate_pool=1800, seed=42):
    """
    V1.3: build many legal candidates, score them inside coherent races, then
    reward portfolios that cover empirically occurring race scripts. No driver
    is forced in/out and there are no manual exposure caps.
    """
    base=optimize_race_sim_lineups(features,sim_matrix,n_lineups=max(250,min(candidate_pool,900)),
                                  salary_cap=salary_cap,roster_size=roster_size,
                                  max_overlap=roster_size,candidate_pool=candidate_pool,
                                  seed=seed,risk_mode="gpp")
    idxs, valid=_candidate_lineup_matrix(base,features)
    if len(valid)==0: return base.head(n_lineups)

    labels=classify_race_scripts(features,sim_matrix)
    scripts=["hamlin_dominant","larson_dominant","split_dominator","chaos_attrition","track_position"]
    probs={s:float(np.mean(labels==s)) for s in scripts}

    # Candidate score distribution from the SAME simulated races.
    scores=np.stack([sim_matrix[:,ix].sum(axis=1) for ix in idxs],axis=1)
    mean=scores.mean(0)
    p90=np.quantile(scores,.90,axis=0)
    p95=np.quantile(scores,.95,axis=0)

    script_scores={}
    for s in scripts:
        mask=labels==s
        if mask.sum() >= 20:
            script_scores[s]=np.quantile(scores[mask],.90,axis=0)
        else:
            script_scores[s]=p90.copy()

    # Normalize within candidate pool so a rare script matters in proportion
    # to how often the simulator actually produces it.
    def z(a):
        sd=np.std(a)
        return (a-np.mean(a))/(sd if sd>1e-9 else 1)
    scenario_component=np.zeros(len(valid))
    for s in scripts:
        scenario_component += probs[s]*z(script_scores[s])

    overall=0.30*z(mean)+0.25*z(p90)+0.20*z(p95)+0.25*scenario_component

    # Greedy diversified portfolio. Overlap rule remains user-controlled.
    chosen=[]
    chosen_sets=[]
    remaining=list(np.argsort(overall)[::-1])
    while remaining and len(chosen)<n_lineups:
        best=None; best_adj=-1e99
        for j in remaining[:500]:
            sset=set(idxs[j].tolist())
            if any(len(sset & old)>max_overlap for old in chosen_sets):
                continue
            # Marginal script coverage bonus: reward a lineup that is strong
            # where already-selected lineups are weaker.
            bonus=0.0
            if chosen:
                for s in scripts:
                    prior=max(script_scores[s][k] for k in chosen)
                    bonus += probs[s]*max(0.0,script_scores[s][j]-prior)
                bonus*=0.015
            adj=overall[j]+bonus
            if adj>best_adj:
                best_adj=adj; best=j
        if best is None: break
        chosen.append(best); chosen_sets.append(set(idxs[best].tolist()))
        remaining.remove(best)

    rows=[]
    for rank,j in enumerate(chosen,1):
        r=base.iloc[valid[j]].copy()
        r["Lineup"]=rank
        r["Mean"]=round(float(mean[j]),2)
        r["P90"]=round(float(p90[j]),2)
        r["P95"]=round(float(p95[j]),2)
        r["ScenarioScore"]=round(float(scenario_component[j]),3)
        for s in scripts:
            r[s+"_P90"]=round(float(script_scores[s][j]),2)
        rows.append(r)
    out=pd.DataFrame(rows)
    out.attrs["script_probabilities"]=probs
    return out

def scenario_summary(features, sim_matrix):
    labels=classify_race_scripts(features,sim_matrix)
    order=["hamlin_dominant","larson_dominant","split_dominator","chaos_attrition","track_position"]
    return pd.DataFrame({
        "Race script":order,
        "Simulation probability":[float(np.mean(labels==s)) for s in order]
    })
