from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Explainability", layout="wide")
st.title("Explainability")

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
IMPORTANCE_CSV = MODELS_DIR / "shap_importance.csv"
SUMMARY_PNG = MODELS_DIR / "shap_summary.png"
LOCAL_CSV = MODELS_DIR / "shap_local_examples.csv"

missing = [p.name for p in (IMPORTANCE_CSV, SUMMARY_PNG, LOCAL_CSV) if not p.exists()]
if missing:
    st.error(
        "Missing explainability artifacts: " + ", ".join(missing) +
        ". Run `python -m src.explain.generate_artifacts` first."
    )
    st.stop()


@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_CSV)


@st.cache_data
def load_local_examples():
    return pd.read_csv(LOCAL_CSV)


importance_df = load_importance()
local_df = load_local_examples()

# --- Global importance ---
st.subheader("Global feature importance")
st.caption(
    "PageValues dominates the ranking by a wide margin over every other feature "
    "— a log scale is used below so the rest remain visible."
)

top_n = importance_df.head(15)
fig = px.bar(
    top_n.sort_values("mean_abs_shap"),
    x="mean_abs_shap",
    y="feature",
    orientation="h",
    log_x=True,
    title="Top 15 features by mean |SHAP value| (log scale)",
)
st.plotly_chart(fig, use_container_width=True)

# --- SHAP summary plot ---
st.subheader("SHAP summary plot")
st.image(str(SUMMARY_PNG), use_container_width=True)

# --- Local explanation demo ---
st.subheader("Local explanation demo")
st.caption("Pick a sample test-set row to see which features pushed its prediction up or down.")

options = local_df["row_index"].tolist()
selected = st.selectbox(
    "Sample row",
    options,
    format_func=lambda idx: (
        f"Row {idx} — predicted {local_df.loc[local_df.row_index == idx, 'predicted_proba'].iloc[0]:.1%}, "
        f"actual {'Purchased' if local_df.loc[local_df.row_index == idx, 'actual_label'].iloc[0] == 1 else 'No purchase'}"
    ),
)

row = local_df[local_df.row_index == selected].iloc[0]
col1, col2 = st.columns(2)
col1.metric("Predicted probability", f"{row['predicted_proba']:.1%}")
col2.metric("Actual outcome", "Purchased" if row["actual_label"] == 1 else "No purchase")

st.write("**Top contributing features:**")
for part in row["top_features"].split("; "):
    feature, value = part.rsplit(": ", 1)
    sign = "🟢 pushes toward purchase" if value.startswith("+") else "🔴 pushes away from purchase"
    st.write(f"- `{feature}`: {value}  — {sign}") 