import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="County Explorer", page_icon="", layout="wide")

# Load data
@st.cache_data
def load_data():
    try:
        analysis = pd.read_csv('data/volatility_final_analysis.csv')
        merged_data = pd.read_csv('data/merged_crop_climate_data.csv')
        return analysis, merged_data
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None

st.title("County-Level Deep Dive")

# Load data
analysis, merged_data = load_data()
if analysis is None:
    st.stop()

# County selection
col1, col2 = st.columns(2)

with col1:
    # Get unique counties
    county_list = analysis.apply(
        lambda x: f"{x['county_name']}, {x['state_name']}", axis=1
    ).unique().tolist()
    county_list.sort()
    
    selected_county_str = st.selectbox(
        "Select County",
        county_list,
        index=0
    )
    
    # Parse selection
    county_name, state_name = selected_county_str.split(", ")

with col2:
    # Get available crops for this county
    county_data = analysis[
        (analysis['county_name'] == county_name) & 
        (analysis['state_name'] == state_name)
    ]
    
    available_crops = county_data['crop'].unique().tolist()
    selected_crop = st.selectbox("Select Crop", available_crops)

# Filter to selected county-crop
selected_data = county_data[county_data['crop'] == selected_crop].iloc[0]

st.markdown("---")

# Key metrics row
col1, col2, col3, col4= st.columns([5, 2, 1, 3])

with col1:
    risk_cat = selected_data['risk_category']
    risk_color = {
        'High Risk': '🔴',
        'Medium Risk': '🟠',
        'Low Risk': '🔵',
        'Improving': '🟢',
        'Insufficient Data': '⚪'
    }.get(risk_cat, '⚪')
    
    st.metric(
        "Risk Category",
        risk_cat,
        delta=risk_color
    )

with col2:
    cv_change = selected_data['yield_cv_change']
    st.metric(
        "Volatility Change",
        f"{cv_change:.2f}%",
        delta=f"{'↑' if cv_change > 0 else '↓'} from baseline"
    )

with col3:
    if 'predicted_cv_change' in selected_data:
        pred_change = selected_data.get('predicted_cv_change', 0)
        st.metric(
            "Predicted Change",
            f"{pred_change:.2f}%",
            delta="Model prediction"
        )

with col4:
    early_cv = selected_data['early_yield_cv']
    late_cv = selected_data['late_yield_cv']
    st.metric(
        "Current Volatility",
        f"{late_cv:.2f}%",
        delta=f"{late_cv - early_cv:.2f}% vs 2005-2014"
    )

st.markdown("---")

# Historical yield trends
if merged_data is not None:
    st.markdown("### Historical Yield Trends")
    
    # Get historical data for this county
    hist_data = merged_data[
        (merged_data['county_name'] == county_name) &
        (merged_data['state_name'] == state_name) &
        (merged_data['crop'] == selected_crop)
    ].sort_values('year')
    
    if len(hist_data) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Yield over time
            fig = px.line(
                hist_data,
                x='year',
                y='yield_value',
                title=f"{selected_crop.capitalize()} Yield Trend (2005-2023)",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Yield (bu/acre)",
                hovermode='x unified'
            )
            fig.add_hline(
                y=hist_data['yield_value'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text="Average"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Calculate rolling volatility
            hist_data['rolling_std'] = hist_data['yield_value'].rolling(window=3, min_periods=1).std()
            
            fig = px.line(
                hist_data,
                x='year',
                y='rolling_std',
                title="3-Year Rolling Volatility",
                markers=True,
                color_discrete_sequence=['#e74c3c']
            )
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Standard Deviation (bu/acre)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Climate trends
st.markdown("### Climate Change Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Temperature")
    temp_change = selected_data['T2M_mean_change']
    temp_std_change = selected_data['T2M_std_change']
    
    st.metric("Avg Temperature Change", f"{temp_change:.2f}°C")
    st.metric("Temperature Variability", f"{temp_std_change:.2f}°C")
    st.metric("Extreme Heat Days", f"+{selected_data['extreme_heat_days_change']:.1f} days")

with col2:
    st.markdown("#### Vegetation Health")
    ndvi_change = selected_data['NDVI_mean_change']
    ndvi_std_change = selected_data['NDVI_std_change']
    
    st.metric("NDVI Change", f"{ndvi_change:.3f}")
    st.metric("NDVI Variability", f"{ndvi_std_change:.3f}")
    st.metric("EVI Change", f"{selected_data['EVI_mean_change']:.3f}")

with col3:
    st.markdown("#### Other Factors")
    st.metric("Humidity Change", f"{selected_data['RH2M_mean_change']:.2f}%")
    st.metric("Solar Radiation", f"{selected_data['ALLSKY_SFC_SW_DWN_mean_change']:.2f}")
    st.metric("Water Stress (NDWI)", f"{selected_data['NDWI_mean_change']:.3f}")

st.markdown("---")

# Feature contribution (if we have it)
st.markdown("### What's Driving Volatility in This County?")

# Create a simple feature importance visualization
features = {
    'Temperature Variability': selected_data['T2M_std_change'],
    'Extreme Heat Days': selected_data['extreme_heat_days_change'],
    'NDVI Variability': selected_data['NDVI_std_change'] * 10,  # Scale for visualization
    'Avg Temperature': selected_data['T2M_mean_change'],
    'Baseline Volatility': selected_data['early_yield_cv'] / 10,  # Scale
}

feature_df = pd.DataFrame({
    'Factor': list(features.keys()),
    'Change': list(features.values())
})
feature_df['Absolute'] = feature_df['Change'].abs()
feature_df = feature_df.sort_values('Absolute', ascending=True)

fig = px.bar(
    feature_df,
    y='Factor',
    x='Change',
    orientation='h',
    title="Relative Changes in Key Risk Factors",
    color='Change',
    color_continuous_scale=['#27ae60', '#f39c12', '#e74c3c'],
    color_continuous_midpoint=0
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)



# Comparison to state average
st.markdown("---")
st.markdown(f"### How Does {county_name} Compare to {state_name}?")

state_avg = analysis[
    (analysis['state_name'] == state_name) &
    (analysis['crop'] == selected_crop)
]['yield_cv_change'].mean()

comparison_data = pd.DataFrame({
    'Location': [county_name, f'{state_name} Average'],
    'Volatility Change': [cv_change, state_avg]
})

fig = px.bar(
    comparison_data,
    x='Location',
    y='Volatility Change',
    color='Volatility Change',
    color_continuous_scale='RdYlGn_r',
    title=f"Volatility Change Comparison - {selected_crop.capitalize()}"
)
st.plotly_chart(fig, use_container_width=True)

if cv_change > state_avg:
    st.warning(f"This county is experiencing **{cv_change - state_avg:.2f}% more** volatility increase than the state average")
else:
    st.success(f"This county is performing **{state_avg - cv_change:.2f}% better** than the state average")