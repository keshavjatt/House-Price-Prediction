import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('models/model.pkl')

# Streamlit page config
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Custom title with Font Awesome icon and colored text
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
        <i class="fa-solid fa-house-chimney" style="color: #27AE60;"></i>
        House Price Prediction App
    </h1>
""", unsafe_allow_html=True)

# Load Font Awesome CDN (add this only once at top of page)
st.markdown("""
    <link rel="stylesheet"
    href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# Form Section
st.write("Fill in the details below to estimate the price of a house.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input("Area (in square feet)", min_value=100.0, step=50.0)
        bedrooms = st.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5])
        stories = st.selectbox("Number of Floors (Stories)", [1, 2, 3, 4])

    with col2:
        bathrooms = st.selectbox("Number of Bathrooms", [1, 2, 3, 4])
        parking = st.selectbox("Parking Spaces Available", [0, 1, 2, 3])

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, parking]],
                              columns=['area', 'bedrooms', 'bathrooms', 'stories', 'parking'])

    prediction = model.predict(input_data)[0]
    formatted_price = f"₹ {prediction:,.2f}"

    st.markdown("---")
    st.subheader("Estimated House Price")
    st.success(formatted_price)
    st.balloons()