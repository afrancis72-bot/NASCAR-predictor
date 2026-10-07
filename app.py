from pathlib import Path
import pandas as pd
import streamlit as st
from nascar_predictor_pro import (
    Config, build_features, simulate, optimize_lineups, portfolio_exposure,
    optimize_scenario_portfolio, scenario_summary, construct_track_dna
)

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="NASCAR Predictor V2.0",layout="wide")
st.title("🏁 NASCAR Predictor V2.0")
st.caption("Track DNA → Driver DNA → Race Scripts → 100K Monte Carlo → DFS Portfolio → Audit")

def csv_upload(label,key):
    f=st.file_uploader(label,type="csv",key=key)
    return pd.read_csv(f) if f else None

def final_grid_ready(updates,field_size):
    if updates is None or not {"Name","qualifying_position"}.issubset(updates.columns):
        return False,"Official qualifying grid not loaded."
    q=pd.to_numeric(updates["qualifying_position"],errors="coerce")
    if len(updates)!=field_size or q.isna().any(): return False,f"Need qualifying positions for all {field_size} drivers."
    if updates["Name"].duplicated().any() or q.duplicated().any(): return False,"Names and starting positions must be unique."
    if set(q.astype(int))!=set(range(1,field_size+1)): return False,f"Starting positions must be 1–{field_size}."
    return True,"Final grid validated."

setup, dna_tab, driver_tab, sim_tab, dfs_tab, audit_tab = st.tabs(
    ["⚙️ Setup / Inputs","🧬 Track DNA","🏎️ Driver DNA","🎲 Race Simulations","💰 DFS Builder","📋 Post-Race Audit"]
)

with setup:
    st.subheader("Weekly race setup")
    st.write("V2.0 no longer reads last week's salaries, priors, projections or lineups from the repository.")
    c1,c2,c3=st.columns(3)
    with c1: dk=csv_upload("1. DraftKings salary CSV","dk")
    with c2: priors=csv_upload("2. Weekly driver priors CSV","priors")
    with c3: track=csv_upload("3. Track profile CSV","track")
    updates=csv_upload("4. Practice / qualifying CSV (optional before qualifying)","updates")
    st.download_button("Download driver-priors template",(ROOT/"driver_priors_template.csv").read_bytes(),"driver_priors_template.csv")
    st.download_button("Download track-profile template",(ROOT/"track_profile_template.csv").read_bytes(),"track_profile_template.csv")
    st.download_button("Download practice/qualifying template",(ROOT/"practice_qualifying_template.csv").read_bytes(),"practice_qualifying_template.csv")
    st.info("Pre-qualifying mode is for research. Final GPP export remains locked until every official starting position is validated.")

if dk is None or priors is None or track is None:
    st.warning("Load the three required weekly files on Setup / Inputs to activate the model.")
    st.stop()

required_track={"track","laps"}
if not required_track.issubset(track.columns):
    st.error("Track profile must contain at least: track, laps.")
    st.stop()

with st.sidebar:
    st.header(str(track.iloc[0]["track"]))
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
track_dna=construct_track_dna(track)

with dna_tab:
    st.subheader(f"{track_dna['track']} — constructed Track DNA")
    st.caption("These are model inputs, not decorative labels. They change the balance of track-type fit, venue history, qualifying/track position and dominator access.")
    a,b,c,d=st.columns(4)
    a.metric("Archetype",track_dna["archetype"].replace("_"," ").title())
    b.metric("Long-run speed",f"{track_dna['long_run_speed']:.0%}")
    c.metric("Track position",f"{track_dna['track_position']:.0%}")
    d.metric("Incident variance",f"{track_dna['incident_variance']:.0%}")
    dna_df=pd.DataFrame([{"Component":k.replace("_"," ").title(),"Importance":v}
                         for k,v in track_dna.items() if isinstance(v,float) and k not in {"track"}])
    st.dataframe(dna_df,use_container_width=True,hide_index=True,
                 column_config={"Importance":st.column_config.ProgressColumn(min_value=0,max_value=1,format="%.0%%")})
    with st.expander("Raw weekly track profile"):
        st.dataframe(track,use_container_width=True,hide_index=True)

with driver_tab:
    st.subheader("Driver DNA / track fit")
    show=["Name","Salary","race_strength","track_dna_fit","dominator_strength","projected_start",
          "data_coverage","model_confidence"]
    st.dataframe(features[show].sort_values("race_strength",ascending=False),use_container_width=True,hide_index=True)
    name=st.selectbox("Inspect driver",features["Name"].tolist())
    r=features.loc[features["Name"]==name].iloc[0]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Race strength",f"{r.race_strength:.2f}")
    c2.metric("Track DNA fit",f"{r.track_dna_fit:.2f}")
    c3.metric("Dominator strength",f"{r.dominator_strength:.2f}")
    c4.metric("Model confidence",f"{r.model_confidence:.0%}")
    st.dataframe(features.loc[features["Name"]==name].T,use_container_width=True)

with sim_tab:
    st.subheader("Race simulation board")
    st.success(f"{int(sims):,} coherent race simulations complete.")
    ss=scenario_summary(proj,sim_matrix)
    ss["Simulation probability"]=(ss["Simulation probability"]*100).round(1).astype(str)+"%"
    st.markdown("#### Race-script mix")
    st.dataframe(ss,use_container_width=True,hide_index=True)
    cols=["Name","Salary","proj_dk","median_p50","floor_p20","ceiling_p90","ceiling_p95","ceiling_p99",
          "sim_top6_pct","win_pct","top5_pct","top10_pct","exp_finish","exp_laps_led","exp_fastest_laps"]
    st.dataframe(proj[cols],use_container_width=True,hide_index=True)
    st.download_button("Download projections",proj.to_csv(index=False).encode(),"nascar_current_race_projections.csv")

with dfs_tab:
    st.subheader("DraftKings portfolio builder")
    if grid_ready: st.success("POST-QUALIFYING / FINAL-GRID MODE — "+grid_message)
    else: st.error("PRE-QUALIFYING MODE — "+grid_message)
    st.caption("Ceiling mode uses coherent race-script outcomes and portfolio exposure/overlap controls. It does not sum six independent player ceilings.")
    if objective=="ceiling" and not grid_ready:
        lineups=pd.DataFrame()
        st.warning("Final GPP construction is locked until the official grid is complete.")
    elif objective=="ceiling":
        lineups=optimize_scenario_portfolio(
            proj,sim_matrix,n_lineups=int(n_lineups),salary_cap=int(salary_cap),roster_size=6,
            max_overlap=int(max_overlap),candidate_pool=2500,seed=int(seed),
            max_exposure=float(max_exposure_pct/100))
    else:
        lineups=optimize_lineups(
            proj,n_lineups=int(n_lineups),salary_cap=int(salary_cap),roster_size=6,
            max_overlap=int(max_overlap),objective=objective,max_exposure=float(max_exposure_pct/100))
    if len(lineups):
        if "Salary" in lineups.columns:
            lineups=lineups[lineups["Salary"]>=int(salary_floor)].reset_index(drop=True)
        st.dataframe(lineups,use_container_width=True,hide_index=True)
        expo=portfolio_exposure(lineups,proj)
        st.markdown("#### Portfolio exposure")
        st.dataframe(expo,use_container_width=True,hide_index=True)
        st.download_button("Download final lineups",lineups.to_csv(index=False).encode(),"nascar_FINAL_lineups.csv")
        st.download_button("Download exposure",expo.to_csv(index=False).encode(),"nascar_FINAL_exposure.csv")
    else:
        st.info("No exportable final portfolio yet.")

with audit_tab:
    st.subheader("Post-race audit")
    st.write("Upload actual race results after the event. V2.0 keeps the audit out of GitHub and generates a downloadable comparison.")
    actual=csv_upload("Actual results CSV (Name, finish; optional dk_points, laps_led, fastest_laps)","actual")
    if actual is not None and {"Name","finish"}.issubset(actual.columns):
        aud=proj.merge(actual,on="Name",how="left",suffixes=("_proj","_actual"))
        aud["finish_error"]=(pd.to_numeric(aud["exp_finish"],errors="coerce")-
                             pd.to_numeric(aud["finish"],errors="coerce")).abs()
        if "dk_points" in aud.columns:
            aud["dk_error"]=(pd.to_numeric(aud["proj_dk"],errors="coerce")-
                             pd.to_numeric(aud["dk_points"],errors="coerce")).abs()
        st.dataframe(aud,use_container_width=True,hide_index=True)
        st.download_button("Download post-race audit",aud.to_csv(index=False).encode(),"nascar_post_race_audit.csv")
