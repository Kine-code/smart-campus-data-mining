import utils.session
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Smart Campus Data Mining",layout="wide")
st.title("Smart Campus - IoT Data Mining Dashboard")
summary=json.loads((ROOT/"results/metrics/summary.json").read_text(encoding="utf-8"))
df=pd.read_csv(ROOT/"data/processed/iot_clean.csv",parse_dates=["timestamp"])
c1,c2,c3,c4=st.columns(4)
c1.metric("Clean records",f"{summary['cleaning']['clean_rows']:,}")
c2.metric("Best clustering K",summary['clustering']['best_k'])
c3.metric("Classification F1",f"{summary['classification']['test_f1']:.3f}")
c4.metric("Regression R²",f"{summary['regression']['test_r2']:.3f}")
st.subheader("Energy by room")
st.bar_chart(df.groupby('room')['energy_kwh'].sum())
st.subheader("Recent anomalies")
an=pd.read_csv(ROOT/"results/metrics/anomaly_records.csv").tail(100)
st.dataframe(an,use_container_width=True)
st.subheader("Model summary")
st.json(summary)
