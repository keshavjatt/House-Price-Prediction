import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# 🚨🚨🚨 IMPORTANT: set_page_config() must be the FIRST Streamlit command!
st.set_page_config(
    page_title="House Price Predictor", 
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Font Awesome CDN (yeh st.markdown hai, set_page_config ke baad allowed hai)
st.markdown("""
    <link rel="stylesheet" 
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    .stButton > button {
        background-color: #2E86C1;
        color: white;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #27AE60;
    }
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Model loading function (caching ke saath)
@st.cache_resource
def load_model():
    """Model ko load kar with proper error handling"""
    try:
        # Current file ka directory find karo
        current_dir = Path(__file__).parent
        model_path = current_dir / 'models' / 'model.pkl'
        
        # Debug info (optional)
        # st.write(f"Looking for model at: {model_path}")
        
        # Check karo model exist karta hai ya nahi
        if not model_path.exists():
            st.error(f"❌ Model not found at {model_path}")
            st.error("Please ensure 'models/model.pkl' exists in your repository")
            return None
            
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Model load karo (yeh function call hai, safe hai set_page_config ke baad)
model = load_model()

# Agar model nahi mila toh app stop karo
if model is None:
    st.stop()

# Custom title with Font Awesome icon
st.markdown("""
    <h1 class='main-header'>
        <i class="fa-solid fa-house-chimney" style="color: #27AE60;"></i>
        House Price Prediction App
    </h1>
""", unsafe_allow_html=True)

# Form Section
st.markdown("### Fill in the details below to estimate the price of a house:")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.number_input(
            "📐 Area (in square feet)", 
            min_value=500.0, 
            max_value=20000.0,
            value=2000.0,
            step=100.0,
            help="Enter the total area in square feet"
        )
        bedrooms = st.selectbox(
            "🛏️ Number of Bedrooms", 
            [1, 2, 3, 4, 5],
            help="Select number of bedrooms"
        )
        stories = st.selectbox(
            "🏢 Number of Floors (Stories)", 
            [1, 2, 3, 4],
            help="Select how many floors"
        )

    with col2:
        bathrooms = st.selectbox(
            "🚿 Number of Bathrooms", 
            [1, 2, 3, 4],
            help="Select number of bathrooms"
        )
        parking = st.selectbox(
            "🅿️ Parking Spaces Available", 
            [0, 1, 2, 3],
            help="Select parking spaces"
        )

    submitted = st.form_submit_button(
        "🔮 Predict Price",
        use_container_width=True
    )

if submitted:
    # Input validation
    if area <= 0:
        st.error("❌ Please enter a valid area")
    else:
        # Create input dataframe
        input_data = pd.DataFrame(
            [[area, bedrooms, bathrooms, stories, parking]],
            columns=['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
        )

        # Predict
        try:
            prediction = model.predict(input_data)[0]
            
            # Format price according to Indian number system
            if prediction >= 10000000:  # 1 Crore+
                formatted_price = f"₹ {prediction/10000000:,.2f} Crore"
            elif prediction >= 100000:  # 1 Lakh+
                formatted_price = f"₹ {prediction/100000:,.2f} Lakh"
            else:
                formatted_price = f"₹ {prediction:,.2f}"

            # Show results with better UI
            st.markdown("---")
            st.subheader("📊 Estimated House Price")
            
            # Create columns for better display
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.metric(
                    label="Predicted Price",
                    value=formatted_price,
                    delta="Estimated Value"
                )
            
            # Show input summary in an expander
            with st.expander("📝 View Input Summary"):
                summary_data = {
                    'Feature': ['Area', 'Bedrooms', 'Bathrooms', 'Stories', 'Parking'],
                    'Value': [f"{area:,.0f} sq.ft", bedrooms, bathrooms, stories, parking]
                }
                summary_df = pd.DataFrame(summary_data)
                st.table(summary_df)
            
            # Success animation
            st.balloons()
            st.success("✅ Prediction completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")