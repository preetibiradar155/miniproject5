
import pickle
import os

import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load the trained model
MODEL_PATH = "house_price_model.pkl"

st.title("🏠 House Price Prediction")
st.write(
    "Predict the estimated price of a house "
    "using a Linear Regression model."
)

if not os.path.exists(MODEL_PATH):
    st.warning(
        "Trained model not found. "
        "Please run train_model.py first."
    )
    st.stop()

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Input form
st.subheader("Enter House Details")

with st.form("house_prediction_form"):
    area = st.number_input(
        "Area (square feet)",
        min_value=100,
        max_value=10000,
        value=1500,
        step=100
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    age = st.number_input(
        "House Age (years)",
        min_value=0,
        max_value=150,
        value=10,
        step=1
    )

    submitted = st.form_submit_button(
        "Predict House Price",
        type="primary",
        use_container_width=True
    )

# Make prediction
if submitted:
    input_data = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Age": [age]
    })

    predicted_price = model.predict(input_data)[0]

    st.divider()
    st.subheader("Prediction Result")

    st.metric(
        label="Estimated House Price",
        value=f"${predicted_price:,.2f}"
    )

    st.caption(
        "This is an estimate from a model trained "
        "on synthetic sample data, not a real "
        "property valuation."
    )

st.divider()
st.caption("Built with Python, Scikit-learn and Streamlit")
