"""Historical rolling-origin NASCAR race-first benchmark (no target-race leakage).

Uses prior Cup race results as coarse proxies for V2.15 inputs, NOT the app's full
practice/qualifying/Reference DNA features. Fastest-lap actuals absent in this feed,
so historical DK accuracy is not evaluated.
"""
import argparse
import numpy as np
import pandas as pd
from nascar_predictor_pro import Config
from nascar_race_engine import simulate


def run(data,season=2025,races=8,sims=1200):
    df=data.copy()
    for c in ["Season","Race","Start","Finish","Laps","Led"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["Season","Race","Start","Finish","Driver"])
    df=df.sort_values(["Season","Race"]).reset_index(drop=True)
    selected=sorted(df.loc[df.Season==season,"Race"].unique())[:races]
    rows=[]
    for race in selected:
        target=df[(df.Season==season)&(df.Race==race)].copy()
        target=target[(target.Start>0)&(target.Finish>0)].sort_values("Start")
        if len(target)<25 or len(target)>40 or target.Start.duplicated().any():continue
        # Prior data ends before target race; never use target finishes/led in inputs.
        history=df[(df.Season<season)|((df.Season==season)&(df.Race<race))]
        hist=history[(history.Season>=season-2)&history.Driver.isin(target.Driver)]
        prior=hist.groupby("Driver").agg(avg_finish=("Finish","mean"),avg_led=("Led","mean"),
                                         starts=("Finish","size"),avg_start=("Start","mean"))
        n=len(target)
        f=target.join(prior,on="Driver")
        observed=f["starts"].fillna(0).to_numpy(float)
        confidence=np.clip(observed/24,.35,.95)
        avg_finish=f["avg_finish"].fillna(n/2).to_numpy(float)
        avg_led=f["avg_led"].fillna(0).to_numpy(float)
        strength=np.clip((n/2-avg_finish)/8,-2.5,2.5)
        dominator=np.clip(np.log1p(avg_led)/2,-1,2.5)
        inp=pd.DataFrame({"Name":target.Driver.to_numpy(),"ID":target.Car.astype(str).to_numpy(),
                          "Salary":np.full(n,8000),"AvgPointsPerGame":np.full(n,30),
                          "projected_start":target.Start.to_numpy(float),
                          "race_strength":strength,"dominator_strength":dominator,
                          "model_confidence":confidence,"audit_dk_prior":strength})
        for c in ["data_coverage","place_diff_room","front_start_score","back_start_score",
                  "audit_vegas","audit_current","audit_intermediate","audit_reference_dna",
                  "audit_reference_form","audit_reference_pd","audit_reference_track",
                  "audit_reference_elo","audit_reference_reliability"]:
            inp[c]=.5
        laps=int(target.Laps.max())
        pred, _=simulate(inp,Config(sims=sims,seed=42+int(race),laps=laps))
        merged=target.merge(pred[["Name","exp_finish","win_pct","exp_laps_led"]],left_on="Driver",right_on="Name")
        actual=merged.Finish.to_numpy(float)
        model=merged.exp_finish.to_numpy(float)
        start=merged.Start.to_numpy(float)
        rows.append({"race":int(race),"track":str(target.Track.iloc[0]),
                     "drivers":len(target),"finish_mae":float(np.mean(abs(model-actual))),
                     "grid_mae":float(np.mean(abs(start-actual))),
                     "laps_led_mae":float(np.mean(abs(merged.exp_laps_led-merged.Led))),
                     "winner_win_probability":float(merged.loc[merged.Finish==1,"win_pct"].iloc[0])})
    if not rows:raise RuntimeError("No historical races were eligible")
    out=pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("\\nAVERAGES:",out[["finish_mae","grid_mae","laps_led_mae","winner_win_probability"]].mean().round(3).to_dict())
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--file",default="cup_series.csv")
    parser.add_argument("--season",type=int,default=2025)
    parser.add_argument("--races",type=int,default=8)
    parser.add_argument("--sims",type=int,default=1200)
    args=parser.parse_args()
    run(pd.read_csv(args.file,dtype={"Car":str}),args.season,args.races,args.sims)
