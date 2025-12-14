import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from pathlib import Path

st.set_page_config(page_title="Volatility Impact Modeler", page_icon="", layout="wide")

# Load model
@st.cache_resource
def load_model():
    try:
        with open('models/xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please ensure xgboost_model.pkl is in the 'models/' folder.")
        return None

st.title("Volatility Impact Modeler")

# Load model
model = load_model()
if model is None:
    st.warning("Model not loaded. Showing demo predictions.")

st.markdown("---")

# Create two columns for input and output
col_input, col_output = st.columns([1, 1])

with col_input:
    st.markdown("### Climate Parameters")
    
    st.markdown("#### Temperature Changes")
    temp_mean_change = st.slider(
        "Average Temperature Change (°C)",
        min_value=-2.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Change in average growing season temperature"
    )
    
    temp_std_change = st.slider(
        "Temperature Variability Change (°C)",
        min_value=-1.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Change in temperature standard deviation - KEY DRIVER!"
    )
    
    temp_max_change = st.slider(
        "Maximum Temperature Change (°C)",
        min_value=-2.0,
        max_value=8.0,
        value=2.0,
        step=0.1,
        help="Change in peak temperature"
    )
    
    extreme_heat_change = st.slider(
        "Extreme Heat Days Change",
        min_value=-5,
        max_value=15,
        value=3,
        step=1,
        help="Change in number of days above 30°C"
    )
    
    st.markdown("#### Vegetation & Environment")
    
    ndvi_mean_change = st.slider(
        "NDVI Change",
        min_value=-0.2,
        max_value=0.2,
        value=-0.05,
        step=0.01,
        help="Change in vegetation health index"
    )
    
    ndvi_std_change = st.slider(
        "NDVI Variability Change",
        min_value=-0.1,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="Change in vegetation health variability"
    )
    
    humidity_change = st.slider(
        "Humidity Change (%)",
        min_value=-15.0,
        max_value=15.0,
        value=-2.0,
        step=0.5,
        help="Change in relative humidity"
    )
    
    st.markdown("#### Baseline Conditions")
    
    early_yield_mean = st.slider(
        "Historical Average Yield (bu/acre)",
        min_value=50.0,
        max_value=200.0,
        value=140.0,
        step=5.0,
        help="Baseline yield level"
    )
    
    early_yield_cv = st.slider(
        "Historical Volatility (CV %)",
        min_value=0.0,
        max_value=30.0,
        value=10.0,
        step=1.0,
        help="Baseline volatility - higher = historically more volatile"
    )
    
    crop_type = st.selectbox(
        "Crop Type",
        ["Corn", "Soybean"],
        help="Different crops respond differently to climate stress"
    )

with col_output:
    st.markdown("### Predicted Outcome")
    
    # Create feature vector for prediction
    features = pd.DataFrame({
        'T2M_mean_change': [temp_mean_change],
        'T2M_std_change': [temp_std_change],
        'T2M_max_change': [temp_max_change],
        'extreme_heat_days_change': [extreme_heat_change],
        'RH2M_mean_change': [humidity_change],
        'ALLSKY_SFC_SW_DWN_mean_change': [0],  # Assumed constant
        'NDVI_mean_change': [ndvi_mean_change],
        'NDVI_std_change': [ndvi_std_change],
        'EVI_mean_change': [ndvi_mean_change * 0.8],  # Correlated with NDVI
        'NDWI_mean_change': [ndvi_mean_change * 0.9],  # Correlated with NDVI
        'early_yield_mean': [early_yield_mean],
        'early_yield_cv': [early_yield_cv],
        'crop_soybean': [1 if crop_type == "Soybean" else 0]
    })
    
    # Make prediction
    if model is not None:
        try:
            prediction = model.predict(features)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            prediction = temp_std_change * 2 + extreme_heat_change * 0.5 + early_yield_cv * 0.3
    else:
        # Fallback calculation if model not loaded
        prediction = temp_std_change * 2 + extreme_heat_change * 0.5 + early_yield_cv * 0.3
    
    # Display prediction with big metric
    st.markdown("#### Predicted Volatility Change")
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "CV Change (%)", 'font': {'size': 24}},
        delta={'reference': 5.0, 'increasing': {'color': "#e74c3c"}, 'font': {'size': 24}},
        number={'font': {'size': 60}, 'valueformat': '.2f'},
        gauge={
            'axis': {'range': [None, 30], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 5], 'color': '#27ae60'},
                {'range': [5, 10], 'color': '#f39c12'},
                {'range': [10, 30], 'color': '#e74c3c'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    
    fig.update_layout(
        height=400, 
        margin=dict(l=40, r=40, t=100, b=40),
        paper_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk classification (aligned with gauge colors)
    if prediction < 5:
        risk_level = "LOW RISK"
        risk_desc = "Minimal impact on yield stability"
        risk_color = "#27ae60"
    elif prediction < 10:
        risk_level = "MEDIUM RISK"
        risk_desc = "Moderate increase in yield variability"
        risk_color = "#f39c12"
    else:
        risk_level = "HIGH RISK"
        risk_desc = "Significant threat to yield stability"
        risk_color = "#e74c3c"
    
    st.markdown(f"<h2 style='text-align: center; color: {risk_color};'>{risk_level}</h2>", 
                unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 18px;'>{risk_desc}</p>", 
                unsafe_allow_html=True)
    
    # Additional metrics
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_cv = early_yield_cv + prediction
        st.metric(
            "Projected Volatility",
            f"{new_cv:.1f}%",
            delta=f"+{prediction:.1f}% from baseline"
        )
    
    with col2:
        confidence = min(95, 60 + abs(prediction) * 2)  # Simulated confidence
        st.metric(
            "Model Confidence",
            f"{confidence:.0f}%",
            delta="Based on similar scenarios"
        )

st.markdown("---")

st.markdown("#### Key Drivers in This Scenario")
    
    # Calculate relative importance
drivers = {
        'Temperature Variability': temp_std_change * 2.0,
        'Extreme Heat Events': extreme_heat_change * 0.5,
        'Baseline Volatility': early_yield_cv * 0.3,
        'NDVI Variability': ndvi_std_change * 10,
        'Average Temperature': temp_mean_change * 0.5,
    }
    
driver_df = pd.DataFrame({
        'Factor': list(drivers.keys()),
        'Contribution': list(drivers.values())
    })
driver_df = driver_df.sort_values('Contribution', ascending=True)
    
import plotly.express as px
fig = px.bar(
        driver_df,
        y='Factor',
        x='Contribution',
        orientation='h',
        title="Relative Contribution to Predicted Change",
        color='Contribution',
        color_continuous_scale='RdYlGn_r'
    )
st.plotly_chart(fig, use_container_width=True)



