import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Crop Yield Volatility Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color palette with high contrast
COLORS = {
    'primary': '#1f77b4',      # Clear blue
    'secondary': '#ff7f0e',    # Orange
    'success': '#2ca02c',      # Green
    'warning': '#ff9800',      # Amber
    'danger': '#d62728',       # Red
    'light_bg': '#f8f9fa',     # Very light gray
    'white': '#ffffff',        # White
    'text_dark': '#212529',    # Almost black
    'text_medium': '#495057',  # Medium gray
    'text_light': '#6c757d',   # Light gray
    'border': '#dee2e6'        # Border
}

# Custom CSS with light theme and high contrast
st.markdown(f"""
    <style>
    /* Force light background everywhere */
    .stApp {{
        background-color: {COLORS['white']};
    }}
    
    .main {{
        background-color: {COLORS['white']};
    }}
    
    /* Header styling */
    .main-header {{
        font-size: 2.8rem;
        font-weight: 700;
        color: {COLORS['text_dark']};
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        margin-bottom: 0;
    }}
    
    .sub-header {{
        font-size: 1.3rem;
        color: {COLORS['text_medium']};
        text-align: center;
        padding-bottom: 2rem;
        font-weight: 400;
    }}
    
    /* Metric cards */
    .metric-container {{
        background-color: {COLORS['light_bg']};
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid {COLORS['border']};
        text-align: center;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {COLORS['primary']};
        margin: 0.5rem 0;
    }}
    
    .metric-label {{
        font-size: 0.95rem;
        color: {COLORS['text_medium']};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metric-delta {{
        font-size: 0.85rem;
        color: {COLORS['text_light']};
        margin-top: 0.5rem;
    }}
    
    /* Section headers */
    .section-header {{
        font-size: 1.8rem;
        font-weight: 600;
        color: {COLORS['text_dark']};
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid {COLORS['primary']};
    }}
    
    /* Info boxes */
    .info-box {{
        background-color: {COLORS['light_bg']};
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid {COLORS['primary']};
        margin: 1rem 0;
        color: {COLORS['text_dark']};
    }}
    
    .info-box h3 {{
        color: {COLORS['text_dark']};
    }}
    
    .info-box p, .info-box ul {{
        color: {COLORS['text_medium']};
    }}
    
    /* Risk badges */
    .risk-badge {{
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }}
    
    .risk-high {{
        background-color: {COLORS['danger']};
        color: white;
    }}
    
    .risk-medium {{
        background-color: {COLORS['warning']};
        color: white;
    }}
    
    .risk-low {{
        background-color: {COLORS['success']};
        color: white;
    }}
    
    /* Navigation cards */
    .nav-card {{
        background-color: {COLORS['white']};
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px solid {COLORS['border']};
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }}
    
    .nav-card:hover {{
        border-color: {COLORS['primary']};
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.15);
    }}
    
    .nav-card h3 {{
        color: {COLORS['text_dark']};
    }}
    
    .nav-card p {{
        color: {COLORS['text_medium']};
    }}
    
    /* Divider */
    hr {{
        margin: 2.5rem 0;
        border: none;
        border-top: 1px solid {COLORS['border']};
    }}
    
    /* Ensure all text is readable */
    p, span, div {{
        color: {COLORS['text_dark']};
    }}
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        predictions = pd.read_csv('data/model_predictions.csv')
        analysis = pd.read_csv('data/volatility_final_analysis.csv')
        return predictions, analysis
    except FileNotFoundError:
        st.error(" Data files not found! Please ensure CSV files are in the 'data/' folder.")
        return None, None

def create_metric_card(label, value, delta, icon="📊"):
    """Create a custom metric card with consistent styling"""
    st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta">{delta}</div>
        </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header"> Crop Yield Volatility Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Climate Change Impact on US Agriculture (2005-2023)</div>', unsafe_allow_html=True)
    
    # Load data
    predictions, analysis = load_data()
    
    if predictions is None or analysis is None:
        st.stop()
    
    # Key metrics row with custom cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_risk_count = len(predictions[predictions['predicted_high_risk'] == True])
        create_metric_card(
            "High-Risk Counties",
            str(high_risk_count),
            "Predicted by Model",
            "⚠️"
        )
    
    with col2:
        total_counties = predictions['county_fp'].nunique()
        create_metric_card(
            "Total Counties",
            str(total_counties),
            "Analyzed Across US",
            "📍"
        )
    
    with col3:
        create_metric_card(
            "Model R² Score",
            "0.566",
            "XGBoost Performance",
            "🎯"
        )
    
    st.markdown("---")
    
    # Introduction section
    st.markdown('<div class="section-header"> Project Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Understanding Agricultural Vulnerability to Climate Change
        
        This dashboard presents a comprehensive analysis of crop yield volatility across US agricultural counties 
        from 2005-2023, examining how climate change is affecting the stability and predictability of corn and 
        soybean production.
        
        ####  Key Findings:
        
        - Temperature variability (not just warming) is the primary driver of increased yield volatility
        - 97 counties identified as high-risk with significant volatility increases
        - Model performance: R² = 0.566 (explains 56.6% of volatility variation)
        - Most counties show stable or improving trends, but vulnerable regions are experiencing deterioration
        
        ####  What is Yield Volatility?
        
        Yield volatility measures how much crop production fluctuates year-to-year. High volatility indicates:
        
        -  Unpredictable income for farmers
        -  Higher insurance costs
        -  Food security risks
        -  Economic instability in agricultural communities
        """)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <h3 style="margin-top: 0; color: {COLORS['text_dark']};"> Statistics</h3>
            <p style="margin: 0.5rem 0; color: {COLORS['text_medium']};"><strong>Dataset Coverage:</strong></p>
            <ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: {COLORS['text_medium']};">
                <li>56,474 observations</li>
                <li>196 agricultural counties</li>
                <li>19 years (2005-2023)</li>
                <li>2 crops (corn, soybean)</li>
            </ul>
            <p style="margin: 1rem 0 0.5rem 0; color: {COLORS['text_medium']};"><strong>Data Sources:</strong></p>
            <ul style="margin: 0.5rem 0; padding-left: 1.5rem; color: {COLORS['text_medium']};">
                <li> NASA POWER (climate)</li>
                <li> MODIS (satellite)</li>
                <li> USDA NASS (yields)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risk Distribution Section - ONE VISUALIZATION PER ROW
    st.markdown('<div class="section-header"> Risk Distribution Analysis (based on actual historical data)</div>', unsafe_allow_html=True)
    
    # First visualization - Pie Chart (full width)
    st.markdown("### Counties by Risk Category")
    
    risk_counts = analysis['risk_category'].value_counts()
    # Define a mapping from risk category to color (use exact names in your CSV)
    risk_colors = {
    "High Risk (Increasing)": COLORS['danger'],         # red
    "Medium Risk (Slight Increase)": COLORS['warning'], # amber
    "Low Risk (Stable)": COLORS['primary'],            # green
    "Improving (Decreasing)": COLORS['success'],       # blue
}

# Map colors safely, fallback to gray if label not in dict
    pie_colors = [risk_colors.get(label, '#95a5a6') for label in risk_counts.index]
 
    fig = go.Figure(data=[go.Pie(
    labels=risk_counts.index,
    values=risk_counts.values,
    hole=0.4,
    marker=dict(
        colors=pie_colors,
        line=dict(color='white', width=2)
    ),
    textposition='auto',
    textinfo='percent',
    textfont=dict(size=14, color='white', family='Arial'),
    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<br><extra></extra>'
)])
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=13, color=COLORS['text_dark'])
        ),
        font=dict(family='Arial', size=13, color=COLORS['text_dark']),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=40, b=100, l=40, r=40),
        annotations=[dict(
            text='Risk<br>Distribution',
            x=0.5, y=0.5,
            font_size=18,
            font_color=COLORS['text_medium'],
            showarrow=False
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics below the chart
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: {COLORS['light_bg']}; border-radius: 8px; border: 1px solid {COLORS['border']};">
            <span class="risk-badge risk-high">High Risk</span>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; color: {COLORS['text_dark']};">{risk_counts.get('High Risk (Increasing)', 0)}</p>
            <p style="margin: 0; color: {COLORS['text_medium']};">counties</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: {COLORS['light_bg']}; border-radius: 8px; border: 1px solid {COLORS['border']};">
            <span class="risk-badge risk-medium">Medium Risk</span>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; color: {COLORS['text_dark']};">{risk_counts.get('Medium Risk (Slight Increase)', 0)}</p>
            <p style="margin: 0; color: {COLORS['text_medium']};">counties</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: {COLORS['light_bg']}; border-radius: 8px; border: 1px solid {COLORS['border']};">
            <span class="risk-badge risk-low">Low Risk</span>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; color: {COLORS['text_dark']};">{risk_counts.get('Low Risk (Stable)', 0)}</p>
            <p style="margin: 0; color: {COLORS['text_medium']};">counties</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Second visualization - Bar Chart (full width)
    st.markdown("### Top 10 Highest Risk Counties")
    
    top_risk = analysis.nlargest(10, 'yield_cv_change')[
        ['county_name', 'state_name', 'crop', 'yield_cv_change']
    ].copy()
    
    top_risk.columns = ['County', 'State', 'Crop', 'CV Change (%)']
    top_risk['CV Change (%)'] = top_risk['CV Change (%)'].round(2)
    top_risk.index = range(1, len(top_risk) + 1)
    
    # Create a horizontal bar chart with proper data viz principles
    fig_bar = go.Figure(data=[
        go.Bar(
            y=top_risk['County'] + ', ' + top_risk['State'] + ' (' + top_risk['Crop'] + ')',
            x=top_risk['CV Change (%)'],
            orientation='h',
            marker=dict(
                color=top_risk['CV Change (%)'],
                colorscale=[
                    [0, COLORS['warning']],
                    [0.5, COLORS['danger']],
                    [1, '#8B0000']
                ],
                showscale=True,
                colorbar=dict(
                    title="CV Change (%)",
                    titleside="right",
                    tickmode="linear",
                    tick0=0,
                    dtick=5,
                    len=0.7
                ),
                line=dict(color=COLORS['border'], width=1)
            ),
            text=top_risk['CV Change (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            textfont=dict(size=12, color=COLORS['text_dark']),
            hovertemplate='<b>%{y}</b><br>' +
                         'CV Change: %{x:.2f}%<br>' +
                         '<extra></extra>'
        )
    ])
    fig_bar.update_layout(
    height=550,
    xaxis_title='Coefficient of Variation Change (%)',
    font=dict(family='Arial', size=12, color=COLORS['text_dark']),
    paper_bgcolor='white',
    plot_bgcolor='white',
    xaxis=dict(
        title=dict(
            text='Coefficient of Variation Change (%)',
            font=dict(size=14, color=COLORS['text_dark'])  # no 'weight'
        ),
        showgrid=True,
        gridcolor='#E5E5E5',
        zeroline=True,
        zerolinecolor='#CCCCCC',
        zerolinewidth=2,
        tickfont=dict(size=11, color=COLORS['text_dark'])
    ),
    yaxis=dict(
        showgrid=False,
        autorange='reversed',
        tickfont=dict(size=11, color=COLORS['text_dark'])
    ),
    margin=dict(t=40, b=80, l=300, r=100)
)

    
    st.plotly_chart(fig_bar, use_container_width=True)
    

    
   
    

if __name__ == "__main__":
    main()