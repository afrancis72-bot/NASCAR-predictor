
from pathlib import Path
import pandas as pd
import streamlit as st
from nascar_predictor_pro import Config, build_features, simulate, optimize_lineups

ROOT=Path(__file__).resolve().parent
DATA=ROOT; OUT=ROOT
st.set_page_config(page_title="NASCAR Predictor V1", layout="wide")
st.title("🏁 NASCAR Predictor V1")
st.caption("Track DNA → Driver Strength → Monte Carlo → DraftKings Projection → Optimizer")

with st.sidebar:
    st.header("Race")
    track_file=DATA/"track_profiles.csv"
    track=pd.read_csv(track_file)
    st.write(track.loc[0,"track"])
    sims=st.selectbox("Simulations",[1000,5000,10000,25000,50000,100000],index=3)
    seed=st.number_input("Seed",value=42,step=1)
    st.header("DFS")
    n_lineups=st.slider("Lineups",1,50,10)
    max_overlap=st.slider("Max shared drivers",0,5,4)
    objective=st.selectbox("Optimizer objective",["ceiling","median","value"])
    salary_cap=st.number_input("Salary cap",value=50000,step=100)

dk_default=pd.read_csv(DATA/"dk_salaries_las_vegas_2026.csv")
pri_default=pd.read_csv(DATA/"driver_priors_las_vegas_2026.csv")

tab1,tab2,tab3,tab4,tab5=st.tabs(["Race Predictor","Driver Explorer","DFS Optimizer","Live Update","Track DNA"])

with tab4:
    st.subheader("Saturday practice / qualifying update")
    st.write("Upload a CSV with Name plus any of: single_lap_rank, avg5_rank, avg10_rank, avg15_rank, avg20_rank, qualifying_position.")
    st.download_button("Download update template",(DATA/"practice_qualifying_template.csv").read_bytes(),
                       file_name="practice_qualifying_template.csv")
    live=st.file_uploader("Practice / qualifying CSV",type="csv")
    updates=pd.read_csv(live) if live else None
    if updates is None:
        st.info("V1A pre-practice mode: live-data weights are redistributed to stable priors. Place differential is provisional.")
    else:
        st.success(f"Loaded live updates for {len(updates)} drivers.")

features=build_features(dk_default,pri_default,track,updates)
cfg=Config(sims=int(sims),seed=int(seed),salary_cap=int(salary_cap),roster_size=6,
           laps=int(track.loc[0,"laps"]),max_overlap=int(max_overlap))
proj,sim_matrix=simulate(features,cfg)

with tab1:
    st.subheader("Projected race / DFS board")
    show=["Name","Salary","proj_dk","ceiling_p90","floor_p20","win_pct","top5_pct","top10_pct",
          "exp_finish","exp_laps_led","exp_fastest_laps","value_per_1k","data_coverage"]
    st.dataframe(proj[show].style.format({
        "proj_dk":"{:.1f}","ceiling_p90":"{:.1f}","floor_p20":"{:.1f}",
        "win_pct":"{:.1%}","top5_pct":"{:.1%}","top10_pct":"{:.1%}",
        "exp_finish":"{:.1f}","exp_laps_led":"{:.1f}","exp_fastest_laps":"{:.1f}",
        "value_per_1k":"{:.2f}"
    }),use_container_width=True)
    st.download_button("Download projections",proj.to_csv(index=False).encode(),
                       file_name="nascar_v1_projections.csv")

with tab2:
    name=st.selectbox("Driver",proj["Name"].tolist())
    r=proj.loc[proj["Name"]==name].iloc[0]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Projected DK",f"{r.proj_dk:.1f}")
    c2.metric("90th percentile",f"{r.ceiling_p90:.1f}")
    c3.metric("Top-5 probability",f"{r.top5_pct:.1%}")
    c4.metric("Expected laps led",f"{r.exp_laps_led:.1f}")
    st.dataframe(features.loc[features["Name"]==name].T,use_container_width=True)

with tab3:
    st.subheader("DraftKings lineup optimizer")
    lineups=optimize_lineups(proj,n_lineups=int(n_lineups),salary_cap=int(salary_cap),
                             roster_size=6,max_overlap=int(max_overlap),objective=objective)
    st.dataframe(lineups,use_container_width=True)
    st.download_button("Download lineups",lineups.to_csv(index=False).encode(),
                       file_name="nascar_v1_lineups.csv")

with tab5:
    st.subheader("Track DNA")
    edited=st.data_editor(track,use_container_width=True,num_rows="fixed")
    st.caption("V1 exposes the DNA weights deliberately. Later versions can learn these from walk-forward backtests by track archetype.")
    st.write("**Current design:** live practice and qualifying weights only activate when those fields exist; otherwise their weight is redistributed to stable priors.")
