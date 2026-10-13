import streamlit as st 
import requests 
import os

st.title("Live Session Scoring") 
st.markdown("Enter session details below to get a real-time purchase prediction from the API.") 

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE_URL}/predict"

with st.form("scoring_form"): 
    col1, col2, col3 = st.columns(3) 

    with col1: 
        st.subheader("Page Activity") 
        administrative = st.number_input("Administrative", min_value = 0, value = 0) 
        administrative_duration = st.number_input("Administrative_Duration", min_value = 0.0, value = 0.0) 
        informational = st.number_input("Informational", min_value = 0, value = 0) 
        informational_duration = st.number_input("Informational_Duration", min_value = 0.0, value = 0.0) 
        product_related = st.number_input("ProductRelated", min_value = 0, value = 1) 
        product_related_duration = st.number_input("ProductRelated_Duration", min_value = 0.0, value = 0.0) 

    with col2: 
        st.subheader("Engagement Metrics") 
        bounce_rates = st.slider("BounceRates", 0.0, 1.0, 0.0) 
        exit_rates = st.slider("ExitRates", 0.0, 1.0, 0.0) 
        page_values = st.number_input("PageValues", min_value = 0.0, value = 0.0) 
        special_day = st.slider("SpecialDay", 0.0, 1.0, 0.0) 

    with col3: 
        st.subheader("Engagement Metrics") 
        month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", 
                                       "Aug", "Sep", "Oct", "Nov", "Dec"]) 
        operating_systems = st.number_input("OperatingSystems", min_value = 1, value = 1) 
        browser = st.number_input("Browser", min_value = 1, value = 1) 
        region = st.number_input("Region", min_value = 1, value = 1) 
        traffic_type = st.number_input("TrafficType", min_value = 1, value = 1) 
        visitor_type = st.selectbox("VisitorType", ["Returning_Visitor", "New_Visitor", "Other"]) 
        weekend = st.checkbox("Weekend") 

    submitted = st.form_submit_button("Predict") 

if submitted: 
    payload = { 
        "Administrative": administrative, 
        "Administrative_Duration": administrative_duration , 
        "Informational": informational, 
        "Informational_Duration": informational_duration, 
        "ProductRelated": product_related, 
        "ProductRelated_Duration": product_related_duration, 
        "BounceRates": bounce_rates, 
        "PageValues": page_values, 
        "ExitRates": exit_rates,
        "SpecialDay": special_day, 
        "Month":month, 
        "OperatingSystems": operating_systems, 
        "Browser": browser, 
        "Region": region, 
        "TrafficType": traffic_type, 
        "VisitorType": visitor_type, 
        "Weekend": weekend,
    } 

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        if response.status_code != 200:
            st.error(f"API returned {response.status_code}: {response.text}")
            st.stop()
        result = response.json() 
         

        st.success("Prediction received") 

        m1, m2, m3 = st.columns(3) 
        m1.metric("Prediction", "Will Purchase" if result["prediction"] == 1 else "Won't Purchase") 
        m2.metric("Probability", f"{result['purchase_probability'] * 100:.2f}%") 
        m3.metric("Confidence", result["confidence"]) 

        st.subheader("Top Contributing Features (SHAP)") 
        st.table(result["top_features"]) 

    except requests.exceptions.ConnectionError: 
        st.error("Could not connet to the API. Make sure it's running: 'uvicorn src.api.main:app --reload'") 

    except Exception as e: 
        st.error(f"Something went wrong: {e}") 
        