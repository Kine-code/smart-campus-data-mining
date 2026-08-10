from pathlib import Path
import pandas as pd
import numpy as np

def preprocess(raw_path=None, out_path=None):
    root=Path(__file__).resolve().parents[1]
    raw_path=Path(raw_path or root/"data/raw/iot_raw.csv")
    out_path=Path(out_path or root/"data/processed/iot_clean.csv")
    df=pd.read_csv(raw_path,parse_dates=["timestamp"])
    initial=len(df)
    duplicates=int(df.duplicated(subset=["device_id","timestamp"]).sum())
    df=df.sort_values(["device_id","timestamp"]).drop_duplicates(subset=["device_id","timestamp"],keep="first")
    missing_before=int(df.isna().sum().sum())
    numeric_cols=["temperature","humidity","light","voltage","current","power"]
    for col in numeric_cols:
        df[col]=df.groupby("device_id")[col].transform(lambda s: s.interpolate(limit_direction="both"))
    # Physically impossible values become NaN then interpolate again
    bounds={"temperature":(-10,70),"humidity":(0,100),"light":(0,3000),"voltage":(180,260),"current":(0,100),"power":(0,20000)}
    invalid=0
    for col,(lo,hi) in bounds.items():
        m=~df[col].between(lo,hi)
        invalid += int(m.sum())
        df.loc[m,col]=np.nan
        df[col]=df.groupby("device_id")[col].transform(lambda s:s.interpolate(limit_direction="both"))
    # Recalculate dependent fields after cleaning
    df["energy_kwh"] = df["power"]/1000*(5/60)
    df["current"] = df["power"]/df["voltage"]
    # Feature engineering
    df["hour"]=df["timestamp"].dt.hour
    df["weekday"]=df["timestamp"].dt.weekday
    df["weekend"]=(df["weekday"]>=5).astype(int)
    df["month"]=df["timestamp"].dt.month
    df["peak_hour"]=df["hour"].between(8,17).astype(int)
    df["temp_humidity_index"] = 0.5*(df["temperature"]+61.0+((df["temperature"]-14.5)*0.8)+(df["humidity"]*0.094))
    df["power_roll_mean_6"] = df.groupby("device_id")["power"].transform(lambda s:s.rolling(6,min_periods=1).mean())
    df["energy_lag1"] = df.groupby("device_id")["energy_kwh"].shift(1)
    df["target_energy_next"] = df.groupby("device_id")["energy_kwh"].shift(-1)
    df["target_activity_next"] = df.groupby("device_id")["activity_state"].shift(-1)
    df=df.dropna(subset=["energy_lag1","target_energy_next","target_activity_next"]).reset_index(drop=True)
    df.to_csv(out_path,index=False)
    summary={"initial_rows":initial,"duplicates_removed":duplicates,"missing_cells_before":missing_before,"invalid_values_corrected":invalid,"clean_rows":len(df)}
    return df,summary

if __name__ == "__main__":
    df,s=preprocess(); print(s)
