from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from utils import get_month_conversion, get_visitor_type_conversion

st.set_page_config(page_title="Purchase Intent Dashboard", layout="wide")

st.title("E-Commerce Purchase Intent Dashboard")
st.markdown("Explainable AI system for predicting customer purchase intent.")

DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent / "dataset" / "ecommerce_sessions.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

st.subheader("Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Sessions", f"{len(df):,}")

with col2:
    conversion_rate = (
        df["Revenue"].mean() * 100
        if "Revenue" in df.columns
        else df["Converted"].mean() * 100
    )
    st.metric("Conversion Rate", f"{conversion_rate:.2f}%")

with col3:
    st.metric("Total Purchases", f"{int(df['Converted'].sum()):,}")

st.subheader("Funnel & Cohort Analytics")

col1, col2 = st.columns(2)

with col1:
    month_data = get_month_conversion(df)
    fig1 = px.bar(
        month_data, x="Month", y="ConversionRate", title="Conversion Rate by Month (%)"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    visitor_data = get_visitor_type_conversion(df)
    fig2 = px.bar(
        visitor_data,
        x="VisitorType",
        y="ConversionRate",
        title="Conversion Rate by Visitor Type (%)",
    )
    st.plotly_chart(fig2, use_container_width=True)
