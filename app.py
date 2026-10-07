from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from nascar_predictor_pro import (
    Config, build_features, simulate, optimize_lineups, portfolio_exposure,
    optimize_scenario_portfolio, scenario_summary, construct_track_dna
)

ROOT=Path(__file__).resolve().parent
CATALOG=pd.read_csv(ROOT/"track_catalog.csv")
st.set_page_config(page_title="NASCAR Predictor V2.2",layout="wide")
st.title("🏁 NASCAR Predictor V2.2")
st.caption("Select Track → Upload DK → Auto Driver DNA → Track DNA → Live Update → 100K Sims → DFS")

def csv_upload(label,key):
    f=st.file_uploader(label,type="csv",key=key)
    return pd.read_csv(f) if f else None

def normalize_dk(dk):
    d=dk.copy()
    aliases={"Name":["Name","Driver","Driver Name"],"ID":["ID","Id","Player ID"],
             "Salary":["Salary"],"AvgPointsPerGame":["AvgPointsPerGame","AvgPoints","FPPG"]}
    for target,opts in aliases.items():
        if target not in d.columns:
            for c in opts:
                if c in d.columns:
                    d[target]=d[c]; break
    if "Name" not in d.columns or "Salary" not in d.columns:
        raise ValueError("DraftKings file needs driver Name and Salary columns.")
    if "ID" not in d.columns: d["ID"]=d["Name"]
    if "AvgPointsPerGame" not in d.columns: d["AvgPointsPerGame"]=0.0
    return d

def auto_priors_from_dk(dk):
    """
    Automatic Driver DNA baseline from fields actually present in DraftKings.
    Salary and FPPG are treated as broad market/performance priors. We do NOT
    fabricate recent speed, track history, comparable-track results or DNF risk.
    """
    salary=pd.to_numeric(dk["Salary"],errors="coerce")
    fppg=pd.to_numeric(dk["AvgPointsPerGame"],errors="coerce")
    sal_pct=salary.rank(pct=True)
    fppg_pct=fppg.rank(pct=True)
    consensus=(0.55*fppg_pct.fillna(.5)+0.45*sal_pct.fillna(.5))*100
    return pd.DataFrame({
        "Name":dk["Name"],
        "manual_speed_rating":consensus,
        "manual_dnf_risk":np.nan,
        "auto_salary_percentile":sal_pct,
        "auto_fppg_percentile":fppg_pct,
        "auto_driver_dna_score":consensus,
        "driver_dna_source":"DK salary + FPPG baseline"
    })

def final_grid_ready(updates,field_size):
    if updates is None or not {"Name","qualifying_position"}.issubset(updates.columns):
        return False,"Official qualifying grid not loaded."
    q=pd.to_numeric(updates["qualifying_position"],errors="coerce")
    if len(updates)!=field_size or q.isna().any(): return False,f"Need qualifying positions for all {field_size} drivers."
    if updates["Name"].duplicated().any() or q.duplicated().any(): return False,"Names and starting positions must be unique."
    if set(q.astype(int))!=set(range(1,field_size+1)): return False,f"Starting positions must be 1–{field_size}."
    return True,"Final grid validated."

setup,dna_tab,driver_tab,live_tab,sim_tab,dfs_tab,audit_tab=st.tabs(
["⚙️ Setup / Inputs","🧬 Track DNA","🏎️ Driver DNA","📡 Live Update","🎲 Race Simulations","💰 DFS Builder","📋 Post-Race Audit"])

with setup:
    st.subheader("Weekly race setup")
    c1,c2=st.columns([1,2])
    with c1:
        selected_track=st.selectbox("1. Select track",CATALOG["track"].tolist())
    track=CATALOG.loc[CATALOG["track"]==selected_track].copy().reset_index(drop=True)
    with c2:
        st.success("Track profile auto-loaded — no weekly track CSV required.")
    dk_raw=csv_upload("2. Upload DraftKings salary CSV","dk")
    st.info("That's the normal weekly setup: choose the track + upload DraftKings. Driver DNA is constructed automatically and the Saturday update lives on its own page.")
    with st.expander("Advanced: optional enriched Driver DNA override"):
        priors_upload=csv_upload("Upload enriched driver-priors CSV","priors")
        st.caption("Optional only. Use this when you have trusted recent-speed, comparable-track, track-history or laps-led data. Missing fields stay neutral; they are never converted to zero.")
        st.download_button("Download enriched-priors template",(ROOT/"driver_priors_template.csv").read_bytes(),"driver_priors_template.csv")

with live_tab:
    st.subheader("Saturday Live Update")
    st.write("Practice and qualifying belong here—not in weekly setup.")
    updates=csv_upload("Upload practice / qualifying CSV","updates")
    st.download_button("Download practice/qualifying template",(ROOT/"practice_qualifying_template.csv").read_bytes(),"practice_qualifying_template.csv")
    if updates is None:
        st.info("PRE-QUALIFYING MODE — research projections are available, but final GPP export stays locked.")
    else:
        st.success(f"Loaded live updates for {len(updates)} drivers. The app will validate the official starting grid before unlocking final GPP builds.")

if dk_raw is None:
    st.warning("Upload the DraftKings salary CSV on Setup / Inputs to activate the model.")
    st.stop()

try:
    dk=normalize_dk(dk_raw)
except Exception as e:
    st.error(str(e)); st.stop()

priors=priors_upload if priors_upload is not None else auto_priors_from_dk(dk)
if "Name" not in priors.columns:
    st.error("Optional priors file must contain Name."); st.stop()

with st.sidebar:
    st.header(selected_track)
    mode="AUTO DRIVER DNA" if priors_upload is None else "ENRICHED DRIVER DNA"
    st.caption(mode)
    sims=st.selectbox("Race simulations",[10000,25000,50000,100000],index=3)
    seed=st.number_input("Seed",value=42,step=1)
    st.divider()
    n_lineups=st.slider("Lineups",1,50,20)
    max_overlap=st.slider("Max shared drivers",0,5,4)
    max_exposure_pct=st.slider("Max driver exposure (%)",10,100,60,5)
    salary_floor=st.number_input("Portfolio salary floor",35000,50000,47000,500)
    salary_cap=st.number_input("Salary cap",45000,50000,50000,100)
    objective=st.selectbox("Build style",["ceiling","median","value"])

grid_ready,grid_message=final_grid_ready(updates,len(dk))
features=build_features(dk,priors,track,updates)
cfg=Config(sims=int(sims),seed=int(seed),salary_cap=int(salary_cap),roster_size=6,
           laps=int(track.iloc[0]["laps"]),max_overlap=int(max_overlap))
proj,sim_matrix=simulate(features,cfg)
dna=construct_track_dna(track)

with dna_tab:
    st.subheader(f"{selected_track} — constructed Track DNA")
    a,b,c,d=st.columns(4)
    a.metric("Archetype",dna["archetype"].replace("_"," ").title())
    b.metric("Long-run speed",f"{dna['long_run_speed']:.0%}")
    c.metric("Track position",f"{dna['track_position']:.0%}")
    d.metric("Incident variance",f"{dna['incident_variance']:.0%}")
    rows=[{"Component":k.replace("_"," ").title(),"Importance":v}
          for k,v in dna.items() if isinstance(v,float)]
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,
        column_config={"Importance":st.column_config.ProgressColumn(min_value=0,max_value=1,format="%.0%%")})
    st.markdown("#### Built-in track profile")
    st.dataframe(track,use_container_width=True,hide_index=True)
    st.caption("The catalog is a starting profile. We will calibrate these values with post-race audits rather than silently changing them week to week.")

with driver_tab:
    st.subheader("Driver DNA")
    if priors_upload is None:
        st.success("AUTO DRIVER DNA — built from the DraftKings salary/FPPG baseline plus this track's DNA.")
        st.caption("Transparency rule: recent speed, comparable-track history, venue history and DNF risk are not invented. Until trusted data is supplied, those components remain neutral and coverage/confidence reflect that.")
    else:
        st.success("ENRICHED DRIVER DNA — trusted override data is active.")
    show=["Name","Salary","race_strength","track_dna_fit","dominator_strength","projected_start","data_coverage","model_confidence"]
    board=features[show].sort_values("race_strength",ascending=False).copy()
    board.insert(0,"DNA Rank",range(1,len(board)+1))
    st.dataframe(board,use_container_width=True,hide_index=True)
    name=st.selectbox("Inspect driver DNA",features["Name"].tolist())
    r=features.loc[features["Name"]==name].iloc[0]
    a,b,c,d=st.columns(4)
    a.metric("Race strength",f"{r.race_strength:.2f}")
    b.metric("Track DNA fit",f"{r.track_dna_fit:.2f}")
    c.metric("Dominator strength",f"{r.dominator_strength:.2f}")
    d.metric("Confidence",f"{r.model_confidence:.0%}")
    st.markdown("#### Signal audit")
    audit_cols=[c for c in ["Name","audit_dk_prior","audit_current","audit_track_type","audit_track_history",
                            "practice_z","projected_start","data_coverage","model_confidence"] if c in features.columns]
    st.dataframe(features.loc[features["Name"]==name,audit_cols].T,use_container_width=True)

with sim_tab:
    st.subheader("Race simulations")
    st.success(f"{int(sims):,} coherent race simulations complete.")
    ss=scenario_summary(proj,sim_matrix)
    ss["Simulation probability"]=(ss["Simulation probability"]*100).round(1).astype(str)+"%"
    st.dataframe(ss,use_container_width=True,hide_index=True)
    cols=["Name","Salary","proj_dk","median_p50","floor_p20","ceiling_p90","ceiling_p95","ceiling_p99",
          "sim_top6_pct","win_pct","top5_pct","top10_pct","exp_finish","exp_laps_led","exp_fastest_laps"]
    st.dataframe(proj[cols],use_container_width=True,hide_index=True)
    st.download_button("Download projections",proj.to_csv(index=False).encode(),"nascar_current_race_projections.csv")

with dfs_tab:
    st.subheader("DraftKings portfolio builder")
    if grid_ready: st.success("FINAL-GRID MODE — "+grid_message)
    else: st.error("PRE-QUALIFYING MODE — "+grid_message)
    if objective=="ceiling" and not grid_ready:
        lineups=pd.DataFrame()
        st.warning("Final GPP portfolio remains locked until the official starting grid is complete.")
    elif objective=="ceiling":
        lineups=optimize_scenario_portfolio(proj,sim_matrix,n_lineups=int(n_lineups),
            salary_cap=int(salary_cap),roster_size=6,max_overlap=int(max_overlap),
            candidate_pool=2500,seed=int(seed),max_exposure=float(max_exposure_pct/100))
    else:
        lineups=optimize_lineups(proj,n_lineups=int(n_lineups),salary_cap=int(salary_cap),
            roster_size=6,max_overlap=int(max_overlap),objective=objective,
            max_exposure=float(max_exposure_pct/100))
    if len(lineups):
        if "Salary" in lineups: lineups=lineups[lineups["Salary"]>=int(salary_floor)].reset_index(drop=True)
        st.dataframe(lineups,use_container_width=True,hide_index=True)
        expo=portfolio_exposure(lineups,proj)
        st.markdown("#### Portfolio exposure")
        st.dataframe(expo,use_container_width=True,hide_index=True)
        st.download_button("Download final lineups",lineups.to_csv(index=False).encode(),"nascar_FINAL_lineups.csv")
        st.download_button("Download exposure",expo.to_csv(index=False).encode(),"nascar_FINAL_exposure.csv")

with audit_tab:
    st.subheader("Post-race audit")
    actual=csv_upload("Actual results CSV (Name, finish; optional dk_points, laps_led, fastest_laps)","actual")
    if actual is not None and {"Name","finish"}.issubset(actual.columns):
        aud=proj.merge(actual,on="Name",how="left",suffixes=("_proj","_actual"))
        aud["finish_error"]=(pd.to_numeric(aud["exp_finish"],errors="coerce")-pd.to_numeric(aud["finish"],errors="coerce")).abs()
        if "dk_points" in aud:
            aud["dk_error"]=(pd.to_numeric(aud["proj_dk"],errors="coerce")-pd.to_numeric(aud["dk_points"],errors="coerce")).abs()
        st.dataframe(aud,use_container_width=True,hide_index=True)
        st.download_button("Download post-race audit",aud.to_csv(index=False).encode(),"nascar_post_race_audit.csv")
