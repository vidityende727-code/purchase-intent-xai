from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")

CSV_PATH = Path(__file__).resolve().parents[3] / "models" / "model_comparison.csv"


@st.cache_data
def load_comparison(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def find_col(cols, name):
    for c in cols:
        if c.lower().replace("-", "_").replace(" ", "_") == name:
            return c
    return None


if not CSV_PATH.exists():
    st.error("model_comparison.csv not found. Run `python -m src.models.train` first.")
    st.stop()

df = load_comparison(CSV_PATH)

model_col = df.select_dtypes(exclude="number").columns[0]
metric_cols = list(df.select_dtypes(include="number").columns)

pr_auc = find_col(metric_cols, "pr_auc")
recall = find_col(metric_cols, "recall")
lead = [c for c in (pr_auc, recall) if c]
rest = [c for c in metric_cols if c not in lead]

df = df[[model_col] + lead + rest].sort_values(lead[0], ascending=False).reset_index(drop=True)

DEPLOYED_MODEL = "CatBoost"

deployed_row = df[df[model_col] == DEPLOYED_MODEL].iloc[0]
st.metric(f"Deployed model ({lead[0]})", DEPLOYED_MODEL, f"{deployed_row[lead[0]]:.3f}")
st.caption(
    f"LightGBM scores marginally higher on {lead[0]}, but CatBoost was selected as the "
    "final model — see model_card.md for the full rationale."
)
DEPLOYED_MODEL = "CatBoost"

st.subheader("Metrics table")
st.dataframe(
    df.style.format({c: "{:.3f}" for c in metric_cols}).highlight_max(subset=lead, color="#1f6f43"),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "Data is imbalanced (~15.7% buyers), so PR-AUC and Recall are the headline metrics here. "
    "Accuracy alone can be misleading on this dataset."
)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(
        px.bar(df, x=model_col, y=lead[0], text=df[lead[0]].round(3), title=f"{lead[0]} by model"),
        use_container_width=True,
    )
with col2:
    if len(lead) > 1:
        st.plotly_chart(
            px.bar(df, x=model_col, y=lead[1], text=df[lead[1]].round(3), title=f"{lead[1]} by model"),
            use_container_width=True,
        ) 