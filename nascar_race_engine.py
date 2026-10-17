"""V2.15 race-first Monte Carlo engine.

Race outcomes are generated first; DraftKings points are derived exclusively from
finish, starting grid, laps led and fastest laps in the SAME simulated race.
This is a segment-level race model, not a telemetry-calibrated lap-by-lap physics model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FINISH_POINTS = np.array([0,45,42,41,40,39,38,37,36,35,34,32,31,30,29,28,27,26,25,24,23,21,20,19,18,17,16,15,14,13,12,10,9,8,7,6,5,4,3,2,1],dtype=float)


def simulate(features, config):
    if not 2 <= len(features) <= 40:
        raise ValueError("Race simulator requires 2 to 40 drivers.")
    if int(config.laps) < 1 or int(config.sims) < 1:
        raise ValueError("Laps and simulation count must be positive.")
    n=len(features); total=int(config.sims); laps=int(config.laps)
    rng=np.random.default_rng(int(config.seed))
    def col(name,default):
        if name not in features:
            return np.full(n,default,dtype=float)
        return pd.to_numeric(features[name],errors="coerce").fillna(default).to_numpy(dtype=float)
    strength=col("race_strength",0)
    dom=col("dominator_strength",0)
    start=col("projected_start",np.nan)
    if not np.all(np.isfinite(start)) or len(set(start.tolist())) != n:
        raise ValueError("Projected starting positions must be complete and unique.")
    confidence=np.clip(col("model_confidence",.45),.35,.95)
    anchor=np.clip(col("audit_dk_prior",0),-2.5,2.5)
    shrink=np.clip((.80-confidence)/.45,0,1)
    strength=(1-.55*shrink)*strength+.55*shrink*anchor
    dom=(1-.65*shrink)*dom+.65*shrink*np.minimum(strength,dom)
    risk=np.clip(col("manual_dnf_risk",.11),.02,.35)
    vol=float(np.clip(col("track_volatility",.95)[0],.35,2.0))
    # Segment count keeps 100k simulations feasible without pretending to have
    # lap-by-lap telemetry. Each segment has a coherent leader and fast-lap mix.
    segment_size=max(3,min(12,round(laps/35)))
    segments=[min(segment_size,laps-i) for i in range(0,laps,segment_size)]
    finish=np.empty((total,n),dtype=np.int16)
    led=np.zeros((total,n),dtype=np.int16)
    fastest=np.zeros((total,n),dtype=np.int16)
    points=np.empty((total,n),dtype=np.float32)
    sigma=np.minimum(1.85-.65*confidence,1.48)
    start_z=-(start-start.mean())/max(float(start.std()),1e-9)
    # Entire-field common race conditions induce correlated pace fluctuations.
    for lo in range(0,total,1024):
        hi=min(total,lo+1024); m=hi-lo
        latent=strength[None,:]+.22*start_z[None,:]+rng.normal(size=(m,n))*sigma[None,:]
        latent+=rng.normal(0,.20*vol,size=(m,1))*dom[None,:]
        incidents=rng.random((m,n))<risk[None,:]
        incident_lap=rng.integers(1,laps+1,size=(m,n))
        # A DNF removes a driver from the lead/fastest-lap pool after its incident.
        # It also pushes the driver toward the rear of the finishing order.
        running=latent.copy()
        row=np.arange(m)
        for k,span in enumerate(segments):
            lap_end=min(laps,(k+1)*segment_size)
            active=~(incidents & (incident_lap<=lap_end))
            # Track-position persistence, plus a race-script shakeup at each segment.
            pace=.55*dom[None,:]+.45*running+rng.normal(0,.32*vol,size=(m,n))
            pace+=.10*start_z[None,:]
            pace=np.where(active,pace,-1e9)
            leader=np.argmax(pace,axis=1)
            led[lo:hi][row,leader]+=span
            # Fastest laps can belong to other competitive cars in the segment.
            # Gumbel race samples a coherent fastest-lap driver for each lap.
            for _ in range(span):
                fast_driver=np.argmax(pace+rng.gumbel(size=(m,n))*.85,axis=1)
                fastest[lo:hi][row,fast_driver]+=1
            # Pit-cycle / caution / tire variation can change relative speed.
            running=.82*running+.18*latent+rng.normal(0,.16*vol,size=(m,n))
        finish_score=running+.12*start_z[None,:]
        # Earlier incidents have a greater finishing penalty than late incidents.
        penalty=2.7+3.0*(laps-incident_lap)/max(laps,1)
        finish_score=np.where(incidents,finish_score-penalty,finish_score)
        order=np.argsort(-finish_score,axis=1)
        f=np.empty((m,n),dtype=np.int16)
        f[row[:,None],order]=np.arange(1,n+1,dtype=np.int16)
        finish[lo:hi]=f
        fp=FINISH_POINTS[np.minimum(f,40)]
        points[lo:hi]=fp+(start[None,:]-f)+.25*led[lo:hi]+.45*fastest[lo:hi]
    assert np.all(led.sum(axis=1)==laps)
    assert np.all(fastest.sum(axis=1)==laps)
    assert np.all(np.sort(finish,axis=1)==np.arange(1,n+1))
    cols=["Name","ID","Salary","AvgPointsPerGame","race_strength","dominator_strength",
          "data_coverage","model_confidence","projected_start","place_diff_room",
          "front_start_score","back_start_score","audit_dk_prior","audit_vegas",
          "audit_current","audit_intermediate","audit_reference_dna",
          "audit_reference_form","audit_reference_pd","audit_reference_track",
          "audit_reference_elo","audit_reference_reliability"]
    out=features[cols].copy()
    out["proj_dk"]=points.mean(axis=0)
    out["floor_p20"]=np.quantile(points,.20,axis=0)
    out["median_p50"]=np.quantile(points,.50,axis=0)
    out["ceiling_p90"]=np.quantile(points,.90,axis=0)
    out["ceiling_p95"]=np.quantile(points,.95,axis=0)
    out["ceiling_p99"]=np.quantile(points,.99,axis=0)
    top=np.argsort(points,axis=1)[:,-min(6,n):]
    out["sim_top6_pct"]=[np.mean(np.any(top==j,axis=1)) for j in range(n)]
    out["win_pct"]=(finish==1).mean(axis=0)
    out["top5_pct"]=(finish<=5).mean(axis=0)
    out["top10_pct"]=(finish<=10).mean(axis=0)
    out["exp_finish"]=finish.mean(axis=0)
    out["exp_laps_led"]=led.mean(axis=0)
    out["exp_fastest_laps"]=fastest.mean(axis=0)
    out["value_per_1k"]=out["proj_dk"]/(out["Salary"]/1000)
    # Keep simulation columns in the same order as sorted projection rows.
    ordering=np.argsort(-out["proj_dk"].to_numpy(),kind="stable")
    return out.iloc[ordering].reset_index(drop=True),points[:,ordering]
