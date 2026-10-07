from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from nascar_auto_intelligence import build_auto_intelligence
from source_diagnostic import run_source_diagnostic
from nascar_reference_dna import build_reference_dna, blend_driver_dna, driver_page, _tables, _flat, parse_profile_text
from nascar_predictor_pro import (
    Config, build_features, simulate, optimize_lineups, portfolio_exposure,
    optimize_scenario_portfolio, scenario_summary, construct_track_dna
)

ROOT=Path(__file__).resolve().parent
CATALOG=pd.read_csv(ROOT/"track_catalog.csv")
st.set_page_config(page_title="NASCAR Predictor V2.10",layout="wide")
st.title("🏁 NASCAR Predictor V2.10")
st.caption("Select Track → Upload DK → NASCAR Reference DNA → Live Update → 100K Sims → DFS")

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

@st.cache_data(ttl=21600,show_spinner=False)
def cached_reference_dna(dk_records,selected_track):
    dk_frame=pd.DataFrame(dk_records)
    ref=build_reference_dna(dk_frame,selected_track)
    return ref,blend_driver_dna(dk_frame,ref)

@st.cache_data(ttl=21600,show_spinner=False)
def cached_auto_intelligence(dk_records,selected_track):
    return build_auto_intelligence(pd.DataFrame(dk_records),selected_track)

def final_grid_ready(updates,field_size):
    if updates is None or not {"Name","qualifying_position"}.issubset(updates.columns):
        return False,"Official qualifying grid not loaded."
    q=pd.to_numeric(updates["qualifying_position"],errors="coerce")
    if len(updates)!=field_size or q.isna().any(): return False,f"Need qualifying positions for all {field_size} drivers."
    if updates["Name"].duplicated().any() or q.duplicated().any(): return False,"Names and starting positions must be unique."
    if set(q.astype(int))!=set(range(1,field_size+1)): return False,f"Starting positions must be 1–{field_size}."
    return True,"Final grid validated."

setup,source_tab,extract_tab,dna_tab,driver_tab,live_tab,sim_tab,dfs_tab,audit_tab=st.tabs(
["⚙️ Setup / Inputs","🔌 Source Diagnostic","🧪 Extraction Diagnostic","🧬 Track DNA","🏎️ Driver DNA","📡 Live Update","🎲 Race Simulations","💰 DFS Builder","📋 Post-Race Audit"])

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

with source_tab:
    st.subheader("V2.4 Data Source Diagnostic")
    st.write("Run this from the deployed Streamlit app. It tests the actual server-to-source connection before we wire a source into Driver DNA.")
    st.caption("Green = reachable from Streamlit. A 401/403 means we do not build against that endpoint. Missing data is never treated as zero.")
    if st.button("Run source diagnostic",type="primary"):
        with st.spinner("Testing NASCAR Feed and NASCAR Reference from this Streamlit server..."):
            diag=run_source_diagnostic()
        st.session_state["source_diag"]=diag
    if "source_diag" in st.session_state:
        diag=st.session_state["source_diag"]
        show=diag[["Source","HTTP","Reachable","Type","Bytes sampled","Seconds"]].copy()
        st.dataframe(show,use_container_width=True,hide_index=True)
        reachable=diag.loc[diag["Reachable"],"Source"].tolist()
        if reachable:
            st.success("Reachable sources: "+", ".join(reachable))
        else:
            st.error("None of the tested sources are reachable from this deployment.")
        with st.expander("Technical details"):
            st.dataframe(diag,use_container_width=True,hide_index=True)
        st.download_button("Download diagnostic CSV",diag.to_csv(index=False).encode(),"nascar_source_diagnostic.csv","text/csv")

if dk_raw is None:
    st.warning("Upload the DraftKings salary CSV on Setup / Inputs to activate the model. Source Diagnostic can be run without DK.")
    st.stop()

try:
    dk=normalize_dk(dk_raw)
except Exception as e:
    st.error(str(e)); st.stop()

baseline=auto_priors_from_dk(dk)
with st.spinner("Building Driver DNA from NASCAR Reference..."):
    try:
        reference_raw,reference_dna=cached_reference_dna(dk.to_dict("records"),selected_track)
        ok=int(reference_raw["reference_ok"].sum())
        intel_status=f"NASCAR Reference loaded for {ok}/{len(reference_raw)} DK drivers."
    except Exception as e:
        reference_raw=pd.DataFrame({"Name":dk["Name"]})
        reference_dna=pd.DataFrame({"Name":dk["Name"]})
        intel_status=f"NASCAR Reference unavailable: {e}"
priors=baseline.merge(reference_dna.drop(columns=["Salary","AvgPointsPerGame"],errors="ignore"),on="Name",how="left")

# V2.10: explicit scored-DNA handoff.  This prevents a stale/duplicate column from
# silently shadowing the Reference score that the engine expects.
_reference_engine_cols=[
    "season_avg_finish","recent_avg_finish","recent_avg_start","recent_place_diff",
    "comparable_track_avg_finish","recent_form_score","place_diff_score",
    "comparable_track_score","track_type_elo_score","reliability_score","driver_dna_score"
]
for _c in _reference_engine_cols:
    if _c in reference_dna.columns:
        _map=reference_dna.set_index("Name")[_c]
        priors[_c]=priors["Name"].map(_map)
# V2.9 integration map: NASCAR Reference fields become generic engine inputs.
# Missing values remain NaN/neutral. Atlanta uses modern superspeedway-style comparable data.
if "comparable_track_avg_finish" in priors.columns:
    priors["track_type_avg_finish"]=pd.to_numeric(priors["comparable_track_avg_finish"],errors="coerce")
# season_avg_finish and recent_avg_finish already use the engine's generic names.
with extract_tab:
    st.subheader("NASCAR Reference — Extraction Diagnostic")
    st.write("This page shows the raw tables returned for one driver and the exact values V2.10 extracted. It does not alter the model.")
    inspect_name=st.selectbox("Driver to inspect",dk["Name"].tolist(),key="extract_driver")
    if st.button("Inspect NASCAR Reference extraction"):
        try:
            html=driver_page(inspect_name)
            parsed=parse_profile_text(html)
            st.markdown("#### V2.10 text-parser output")
            _diag={k:v for k,v in parsed.items() if k!="recent_rows"}
            st.dataframe(pd.DataFrame({"metric":list(_diag.keys()),"value":list(_diag.values())}),use_container_width=True,hide_index=True)
            if parsed.get("recent_rows"):
                st.markdown("#### Parsed recent Cup races")
                st.dataframe(pd.DataFrame(parsed["recent_rows"]),use_container_width=True,hide_index=True)
            tabs=[_flat(t) for t in _tables(html)]
            st.success(f"Retrieved {len(html):,} HTML characters and found {len(tabs)} HTML tables.")
            if not tabs:
                st.error("No HTML tables were parsed from this driver page.")
            for i,t in enumerate(tabs):
                with st.expander(f"Raw table {i+1} — {len(t)} rows × {len(t.columns)} columns"):
                    st.write("Columns:",list(t.columns))
                    st.dataframe(t.head(20),use_container_width=True,hide_index=True)
            with st.expander("HTML text sample"):
                from bs4 import BeautifulSoup
                txt=" ".join(BeautifulSoup(html,"html.parser").stripped_strings)
                st.text(txt[:12000])
        except Exception as e:
            st.exception(e)

if priors_upload is not None:
    if "Name" not in priors_upload.columns:
        st.error("Optional priors file must contain Name."); st.stop()
    over=priors_upload.set_index("Name")
    priors=priors.set_index("Name")
    for c in over.columns:
        prior_col=priors[c] if c in priors.columns else pd.Series(index=priors.index,dtype=float)
        priors[c]=over[c].combine_first(prior_col)
    priors=priors.reset_index()

with st.sidebar:
    st.header(selected_track)
    mode="NASCAR REFERENCE DNA" if priors_upload is None else "REFERENCE + MANUAL OVERRIDE"
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

# V2.10 safety gate: a working Reference integration must create cross-driver spread.
_ref_audit_cols=["audit_reference_form","audit_reference_pd","audit_reference_track",
                 "audit_reference_elo","audit_reference_reliability"]
_ref_present=[c for c in _ref_audit_cols if c in features.columns]
_ref_spread={}
for _c in _ref_present:
    _v=pd.to_numeric(features[_c],errors="coerce")
    _sd=_v.std(skipna=True)
    _ref_spread[_c]=0.0 if pd.isna(_sd) else float(_sd)
_reference_integration_ok=bool(_ref_present) and any(v>1e-8 for v in _ref_spread.values())
# V2.8: Reference DNA is authoritative for coverage/confidence.
if {"data_coverage_pct","model_confidence"}.issubset(reference_dna.columns):
    _cov=reference_dna[["Name","data_coverage_pct","model_confidence"]].copy()
    # priors may already have these V2.6 columns, so build_features can carry them through.
    # Prefer the authoritative Reference values without creating _x/_y collisions.
    features=features.drop(columns=["data_coverage","data_coverage_pct","model_confidence"],errors="ignore")
    features=features.merge(_cov,on="Name",how="left")
    features["data_coverage"]=pd.to_numeric(features["data_coverage_pct"],errors="coerce").fillna(15.0)/100.0
    # Simulation engine requires numeric confidence. Keep the human-readable label separately.
    features["model_confidence_label"]=features["model_confidence"].astype(str)
    _confidence_map={"HIGH":0.90,"MEDIUM":0.70,"LOW":0.45}
    features["model_confidence"]=features["model_confidence_label"].str.upper().map(_confidence_map).fillna(0.45).astype(float)

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
    if "loaded" in intel_status.lower():
        st.success("NASCAR REFERENCE DRIVER DNA — "+intel_status)
    else:
        st.warning(intel_status+" DK baseline remains active; missing web data is neutral.")
    if priors_upload is None:
        st.success("AUTO DRIVER DNA — NASCAR Reference recent form and track-history signals are blended with the DK market baseline and Track DNA.")
        st.caption("Transparency rule: missing NASCAR Reference components are omitted and the remaining weights renormalize. Missing data never scores as zero.")
    else:
        st.success("ENRICHED DRIVER DNA — trusted override data is active.")
    show=["Name","Salary","race_strength","track_dna_fit","dominator_strength","projected_start","data_coverage","model_confidence"]
    board=features[show].sort_values("race_strength",ascending=False).copy()
    if updates is None and "projected_start" in board.columns:
        board["projected_start"]="PRE-QUAL"
    board.insert(0,"DNA Rank",range(1,len(board)+1))
    st.dataframe(board,use_container_width=True,hide_index=True)
    st.markdown("#### Driver DNA components")
    dna_cols=["Name","driver_dna_score","recent_form_score","place_diff_score","comparable_track_score","track_type_elo_score","reliability_score","dk_market_score","data_coverage_pct","model_confidence"]
    available=[c for c in dna_cols if c in reference_dna.columns]
    if available:
        st.dataframe(reference_dna[available].sort_values("driver_dna_score",ascending=False),use_container_width=True,hide_index=True)

    st.markdown("#### Model integration audit")
    if _reference_integration_ok:
        st.success("PASS — NASCAR Reference DNA is entering the race model.")
    else:
        st.error("FAIL — NASCAR Reference DNA has no cross-field model spread. Do not trust final simulations.")
    st.caption("These are the normalized NASCAR Reference signals actually entering race strength / Track DNA fit. Positive and negative values across drivers confirm integration.")
    integration_cols=["Name","audit_reference_form","audit_reference_pd","audit_reference_track",
                      "audit_reference_elo","audit_reference_reliability","audit_current",
                      "audit_track_type","race_strength","track_dna_fit","dominator_strength"]
    integration_cols=[c for c in integration_cols if c in features.columns]
    st.dataframe(features[integration_cols].sort_values("race_strength",ascending=False),
                 use_container_width=True,hide_index=True)
    if _ref_spread:
        st.caption("Cross-field SD — " + " | ".join(
            f"{k.replace('audit_reference_','')}: {v:.3f}" for k,v in _ref_spread.items()))
    name=st.selectbox("Inspect driver DNA",features["Name"].tolist())
    r=features.loc[features["Name"]==name].iloc[0]
    a,b,c,d=st.columns(4)
    a.metric("Race strength",f"{r.race_strength:.2f}")
    b.metric("Track DNA fit",f"{r.track_dna_fit:.2f}")
    c.metric("Dominator strength",f"{r.dominator_strength:.2f}")
    d.metric("Confidence",f"{r.model_confidence:.0%}")
    with st.expander("Automated NASCAR data for this driver"):
        st.dataframe(reference_dna.loc[reference_dna["Name"]==name].T,use_container_width=True)
    st.markdown("#### Signal audit")
    audit_cols=[c for c in ["Name","audit_dk_prior","audit_reference_form","audit_reference_pd",
                            "audit_reference_track","audit_reference_elo","audit_reference_reliability",
                            "audit_current","audit_track_type","audit_track_history","practice_z",
                            "projected_start","data_coverage","model_confidence"] if c in features.columns]
    st.dataframe(features.loc[features["Name"]==name,audit_cols].T,use_container_width=True)

coverage_ready = ("data_coverage_pct" in reference_dna.columns and reference_dna["data_coverage_pct"].median() >= 55)

with sim_tab:
    st.subheader("Race simulations")
    if not _reference_integration_ok:
        st.error("REFERENCE DNA SAFETY CHECK FAILED. Simulation output is diagnostic only; do not use it for final DFS decisions.")
    else:
        st.success(f"{int(sims):,} coherent race simulations complete with NASCAR Reference DNA integrated.")
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
