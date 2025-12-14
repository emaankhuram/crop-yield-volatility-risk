import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Analytics", page_icon="", layout="wide")

# Load data
@st.cache_data
def load_data():
    try:
        analysis = pd.read_csv('data/volatility_final_analysis.csv')
        feature_imp = pd.read_csv('data/feature_importance.csv')
        return analysis, feature_imp
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None

st.title("Analysis & Insights")

# Load data
analysis, feature_imp = load_data()
if analysis is None:
    st.stop()

# Key findings banner
st.info("""
**Key Findings:**
- Temperature variability (not just warming) is the #1 driver of yield volatility
- 97 counties identified as high-risk with CV increases >5%
- Geographic heterogeneity: impacts concentrated in marginal agricultural regions
""")

st.markdown("---")

# Feature Importance
st.markdown("## What Drives Yield Volatility?")

col1, col2 = st.columns([2, 1])

with col1:
    if feature_imp is not None and len(feature_imp) > 0:
        # Plot top 15 features
        top_features = feature_imp.head(15).copy()
        
        # Clean feature names
        top_features['Feature_Clean'] = top_features['Feature'].str.replace('_', ' ').str.title()
        
        # Reverse order so most important is at top
        top_features = top_features.iloc[::-1]
        
        fig = px.bar(
            top_features,
            y='Feature_Clean',
            x='RF_Importance',
            orientation='h',
            title="Top 15 Feature Importance (Random Forest)",
            labels={'RF_Importance': 'Importance Score', 'Feature_Clean': 'Feature'},
            color='RF_Importance',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Feature importance data not available")

with col2:
    st.markdown("### Key Takeaways")
    st.markdown("""
    **1. Baseline Matters Most (65%)**
    - Counties that were volatile stay volatile
    
    **2. Temperature Variability (4-6%)**
    - Erratic weather > gradual warming
    
    **3. Extreme Events (2-6%)**
    - Heat waves, frost events
    
    **4. Vegetation Response (4-5%)**
    - NDVI variability captures stress
    """)

st.markdown("---")

# Correlation Analysis
st.markdown("### Climate-Volatility Relationships")

col1, col2 = st.columns(2)

with col1:
    # Temperature variability vs volatility change
    fig = px.scatter(
        analysis,
        x='T2M_std_change',
        y='yield_cv_change',
        color='crop',
        trendline='ols',
        title="Temperature Variability vs Yield Volatility",
        labels={
            'T2M_std_change': 'Temperature Variability Change (°C)',
            'yield_cv_change': 'Yield Volatility Change (%)'
        },
        hover_data=['county_name', 'state_name']
    )
    fig.add_annotation(
        text=f"Correlation: {analysis['T2M_std_change'].corr(analysis['yield_cv_change']):.3f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98, showarrow=False,
        bgcolor="white", bordercolor="black", borderwidth=1
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Extreme heat vs volatility
    fig = px.scatter(
        analysis,
        x='extreme_heat_days_change',
        y='yield_cv_change',
        color='crop',
        trendline='ols',
        title="Extreme Heat Days vs Yield Volatility",
        labels={
            'extreme_heat_days_change': 'Change in Extreme Heat Days',
            'yield_cv_change': 'Yield Volatility Change (%)'
        },
        hover_data=['county_name', 'state_name']
    )
    fig.add_annotation(
        text=f"Correlation: {analysis['extreme_heat_days_change'].corr(analysis['yield_cv_change']):.3f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98, showarrow=False,
        bgcolor="white", bordercolor="black", borderwidth=1
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# Geographic patterns
st.markdown("## Geographic Patterns")

# Top and bottom states
state_summary = analysis.groupby('state_name').agg({
    'yield_cv_change': 'mean',
    'county_fp': 'count'
}).reset_index()
state_summary.columns = ['State', 'Avg CV Change', 'Counties']
state_summary = state_summary[state_summary['Counties'] >= 3]  # At least 3 counties

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Most Vulnerable States")
    top_states = state_summary.nlargest(10, 'Avg CV Change')
    
    fig = px.bar(
        top_states,
        x='Avg CV Change',
        y='State',
        orientation='h',
        title="Top 10 States with Highest Volatility Increase",
        color='Avg CV Change',
        color_continuous_scale='Reds'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Most Resilient States")
    bottom_states = state_summary.nsmallest(10, 'Avg CV Change')
    
    fig = px.bar(
        bottom_states,
        x='Avg CV Change',
        y='State',
        orientation='h',
        title="Top 10 States with Lowest Volatility Change",
        color='Avg CV Change',
        color_continuous_scale='Greens_r'
    )
    fig.update_layout(yaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Climate change indicators
st.markdown("## Climate Change Indicators Across All Counties")

climate_vars = ['T2M_mean_change', 'T2M_std_change', 'extreme_heat_days_change', 
                'NDVI_mean_change', 'NDVI_std_change']

# Create distribution plots
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        'Temp Change', 'Temp Variability', 'Extreme Heat Days',
        'NDVI Change', 'NDVI Variability', 'Humidity Change'
    )
)

positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
vars_to_plot = ['T2M_mean_change', 'T2M_std_change', 'extreme_heat_days_change',
                'NDVI_mean_change', 'NDVI_std_change', 'RH2M_mean_change']

for (row, col), var in zip(positions, vars_to_plot):
    fig.add_trace(
        go.Histogram(x=analysis[var], name=var, showlegend=False),
        row=row, col=col
    )

fig.update_layout(height=600, title_text="Distribution of Climate Change Indicators")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


