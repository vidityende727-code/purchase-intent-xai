import requests
import streamlit as st
import os

st.set_page_config(page_title="System", layout="wide")
st.title("System")

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.subheader("API health")

try:
    response = requests.get(f"{API_URL}/health", timeout=3)
    response.raise_for_status()
    data = response.json()

    col1, col2, col3 = st.columns(3)
    col1.metric("API status", data.get("status", "unknown").upper())
    col2.metric("Deployed model", data.get("model_name", "—"))
    col3.metric("Decision threshold", f"{data.get('threshold', 0):.2f}")

    st.success("FastAPI service is reachable.")
except requests.exceptions.RequestException:
    st.error(
        "Could not reach the FastAPI service at "
        f"`{API_URL}`. Make sure it's running: `uvicorn src.api.main:app --reload`."
    )