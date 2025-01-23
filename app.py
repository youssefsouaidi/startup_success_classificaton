import streamlit as st
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


xg_model = joblib.load(r"C:\Users\Souai\PycharmProjects\challenge\.venv\xgfinal_model.joblib")
rf_model = joblib.load(r"C:\Users\Souai\PycharmProjects\challenge\.venv\best_rf_model.joblib")

st.title("Startup Success Prediction")
st.write("Enter the following parameters to predict success metrics using two different models.")

with st.form("input_form"):
    funding_rounds = st.number_input("Funding Rounds",)
    total_funding = st.number_input("Total Funding ($M)",  )
    team_size = st.number_input("Team Size",)
    tech_stack_size = st.number_input("Tech Stack Size")
    patents = st.number_input("Patents",)
    burned_rate = st.number_input("Burned Rate ($K/month)")
    revenue_growth = st.number_input("Revenue Growth (%)")
    market_size = st.number_input("Market Size ($M)")
    competitors = st.number_input("Competitors")
    social_media_score = st.number_input("Social Media Score")
    client_retention = st.number_input("Client Retention (%)")
    pivot_count = st.number_input("Pivot Count")
    regulatory_score = st.number_input("Regulatory Score")

    submitted = st.form_submit_button("Predict")

if submitted:
    required_columns = [
        "funding_rounds", "total_funding", "team_size", "tech_stack_size", "patents",
        "burned_rate", "revenue_growth", "market_size", "competitors",
        "social_media_score", "pivot_count", "regulatory_score", "client_retention"
    ]

    input_data = pd.DataFrame(
        [[
            funding_rounds, total_funding, team_size, tech_stack_size, patents,
            burned_rate, revenue_growth, market_size, competitors, social_media_score,
            pivot_count, regulatory_score, client_retention
        ]],
        columns=required_columns
    )

    # Convert input data to DMatrix for XGBoost
    dmatrix_input = xgb.DMatrix(input_data)

    # Predict using models
    xg_prediction = xg_model.predict(dmatrix_input)[0]
    rf_prediction = rf_model.predict(input_data.values)[0]

    # Define interpretation
    def interpret_prediction(prediction):
        if prediction == 0:
            return "Failure"
        elif prediction == 1:
            return "Success"
        elif prediction == 2:
            return "Acquisition"
        else:
            return "Unknown"

    xg_interpretation = interpret_prediction(xg_prediction)
    rf_interpretation = interpret_prediction(rf_prediction)

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**XGBoost Model Prediction:** {xg_prediction} ({xg_interpretation})")
    st.write(f"**Random Forest Model Prediction:** {rf_prediction} ({rf_interpretation})")

    # Add some styling
    st.markdown("""
    <style>
        .stButton > button {background-color: #4CAF50; color: white; font-size: 16px;}
        .stMarkdown > div {color: #000000; font-size: 18px; margin-top: 20px;}
    </style>
    """, unsafe_allow_html=True)
