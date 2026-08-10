from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

ROOMS = [
    ("A",1,"A101","Classroom"),("A",1,"A102","Classroom"),("A",2,"A201","Classroom"),
    ("B",1,"B101","Classroom"),("B",2,"B201","Classroom"),("B",2,"B202","Classroom"),
    ("C",1,"C101","Office"),("C",2,"C201","Lab"),("D",1,"D101","Lab"),("D",2,"D201","ServerRoom")
]

def occupancy_pattern(ts, room_type):
    h, wd = ts.hour + ts.minute/60, ts.weekday()
    if room_type == "ServerRoom":
        return 1
    if wd >= 5:
        base = 0.12 if room_type != "Office" else 0.05
    else:
        if room_type == "Classroom":
            base = 0.86 if (7.5 <= h <= 11.5 or 13 <= h <= 17.5) else 0.08
        elif room_type == "Lab":
            base = 0.92 if 8 <= h <= 18.5 else 0.12
        else:
            base = 0.88 if 8 <= h <= 17 else 0.05
    return int(rng.random() < base)

def main():
    out = Path(__file__).resolve().parents[1] / "data" / "raw" / "iot_raw.csv"
    times = pd.date_range("2026-06-01", "2026-06-30 23:55", freq="5min")
    rows=[]
    device_idx=1
    for building,floor,room,room_type in ROOMS:
        device_id=f"D{device_idx:03d}"; device_idx+=1
        for ts in times:
            occ=occupancy_pattern(ts, room_type)
            hour=ts.hour+ts.minute/60
            ambient = 26.5 + 3.8*np.sin((hour-9)/24*2*np.pi) + rng.normal(0,0.45)
            if room_type == "ServerRoom": ambient += 1.2
            temp = ambient + (0.6 if occ else 0) + rng.normal(0,0.25)
            humidity = 70 - (temp-25)*2.4 + rng.normal(0,2.2)
            daylight = max(0, 900*np.sin((hour-6)/12*np.pi)) if 6 <= hour <= 18 else 0
            light = daylight*0.35 + (420 if occ else 15) + rng.normal(0,25)
            if room_type == "ServerRoom": light = 120 + rng.normal(0,15)
            voltage = 220 + rng.normal(0,2.2)
            base_power = {"Classroom":90,"Office":130,"Lab":230,"ServerRoom":1800}[room_type]
            active_extra = {"Classroom":780,"Office":520,"Lab":1600,"ServerRoom":500}[room_type]
            power = base_power + occ*active_extra + max(0,temp-28)*55 + rng.normal(0,45 if room_type!="ServerRoom" else 90)
            power = max(20,power)
            current = power/voltage + rng.normal(0,0.03)
            energy = power/1000*(5/60)
            status = "ON" if power > base_power*1.25 else "OFF"
            activity = "HIGH" if power > base_power + active_extra*0.75 else ("NORMAL" if occ else "EMPTY")
            rows.append([ts,device_id,building,floor,room,room_type,temp,humidity,light,voltage,current,power,energy,occ,status,activity,0])
    df=pd.DataFrame(rows,columns=["timestamp","device_id","building","floor","room","room_type","temperature","humidity","light","voltage","current","power","energy_kwh","occupancy","status","activity_state","injected_anomaly"])

    # Inject realistic anomalies (~0.3%)
    n=len(df); anomaly_idx=rng.choice(n,size=int(n*0.003),replace=False)
    thirds=np.array_split(anomaly_idx,3)
    df.loc[thirds[0],"power"] *= rng.uniform(3.5,7.0,size=len(thirds[0]))
    df.loc[thirds[0],"current"] = df.loc[thirds[0],"power"]/df.loc[thirds[0],"voltage"]
    df.loc[thirds[0],"energy_kwh"] = df.loc[thirds[0],"power"]/1000*(5/60)
    df.loc[thirds[1],"temperature"] += rng.uniform(12,30,size=len(thirds[1]))
    df.loc[thirds[2],"humidity"] = rng.choice([-8, 130], size=len(thirds[2]))
    df.loc[anomaly_idx,"injected_anomaly"]=1

    # Missing values (~0.2%) excluding identifiers/target labels
    for col in ["temperature","humidity","light","voltage","current","power"]:
        idx=rng.choice(n,size=int(n*0.002),replace=False)
        df.loc[idx,col]=np.nan

    # Duplicate ~0.25%
    dup=df.sample(frac=0.0025,random_state=SEED)
    df=pd.concat([df,dup],ignore_index=True).sample(frac=1,random_state=SEED).reset_index(drop=True)
    df.to_csv(out,index=False)
    print(f"Generated {len(df):,} rows -> {out}")

if __name__ == "__main__": main()
