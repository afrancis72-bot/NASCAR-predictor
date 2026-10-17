"""Convert open 2014-2024 NASCAR Cup result dataset into backtest schema."""
import pandas as pd
src=pd.read_csv("cup_race_results_2014_2024.csv",dtype={"car_number":str})
src=src.sort_values(["race_season","race_date","race_id"])
src["Race"]=src.groupby("race_season")["race_id"].transform(lambda s: pd.factorize(s)[0]+1)
out=src.rename(columns={"race_season":"Season","track_name":"Track","starting_position":"Start",
"finishing_position":"Finish","driver_fullname":"Driver","car_number":"Car",
"actual_laps":"Laps","laps_led":"Led"})
out[["Season","Race","Track","Start","Finish","Driver","Car","Laps","Led"]].to_csv("cup_series.csv",index=False)
print("Converted",len(out),"driver-race rows; seasons",out.Season.min(),out.Season.max())
