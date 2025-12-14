import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import folium
from streamlit_folium import st_folium
import json

st.set_page_config(page_title="Risk Map", page_icon="", layout="wide")

# Load data
@st.cache_data
def load_data():
    try:
        predictions = pd.read_csv('data/model_predictions.csv')
        return predictions
    except FileNotFoundError:
        st.error("Data file not found!")
        return None

# Load US counties GeoJSON
@st.cache_data
def load_geojson():
    """Load US counties GeoJSON from online source"""
    try:
        url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        response = requests.get(url)
        return response.json()
    except Exception as e:
        st.warning(f"Could not load GeoJSON: {e}")
        return None

st.title("Geographic Risk Distribution")

# Load data
predictions = load_data()
if predictions is None:
    st.stop()

counties_geojson = load_geojson()

# Use all data without filters
filtered_data = predictions.copy()

# Classify risk
filtered_data['risk_level'] = pd.cut(
    filtered_data['predicted_cv_change'],
    bins=[-float('inf'), 0, 2, 5, float('inf')],
    labels=['Improving', 'Low Risk', 'Medium Risk', 'High Risk']
)

# Create FIPS code for choropleth matching (5 digits: state + county)
filtered_data['fips'] = filtered_data['state_fp'].astype(str).str.zfill(2) + \
                        filtered_data['county_fp'].astype(str).str.zfill(3)

# Create FIPS code (5 digits: state + county)
filtered_data['fips'] = filtered_data['state_fp'].astype(str).str.zfill(2) + \
                        filtered_data['county_fp'].astype(str).str.zfill(3)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Use the same predicted_high_risk column as app.py
    if 'predicted_high_risk' in filtered_data.columns:
        high_risk = (filtered_data['predicted_high_risk'] == True).sum()
    else:
        high_risk = len(filtered_data[filtered_data['predicted_cv_change'] > 5])
    st.metric("High-Risk Counties", high_risk)

with col2:
    medium_risk = len(filtered_data[(filtered_data['predicted_cv_change'] >= 2) & (filtered_data['predicted_cv_change'] <= 5)])
    st.metric("Medium Risk (2-5%)", medium_risk)

with col3:
    low_risk = len(filtered_data[filtered_data['predicted_cv_change'] < 2])
    st.metric("Low Risk (<2%)", low_risk)

with col4:
    avg_pred = filtered_data['predicted_cv_change'].mean()
    st.metric("Avg CV Change", f"{avg_pred:.2f}%")

st.markdown("---")

# Main visualization
st.markdown("### Predicted County Risk Choropleth Map")

if counties_geojson is not None:
    # Aggregate by county (in case multiple crops per county)
    county_agg = filtered_data.groupby('fips').agg({
        'predicted_cv_change': 'mean',
        'county_name': 'first',
        'state_name': 'first',
        'crop': lambda x: ', '.join(x.unique())
    }).reset_index()
    
    # Create colormap function
    def get_color(cv_change):
        if cv_change < 0:
            return '#27ae60'  # Green (improving)
        elif cv_change < 2:
            return '#FFD700'  # Yellow (low risk)
        elif cv_change < 5:
            return '#FF8C00'  # Dark orange/red-orange (medium risk)
        else:
            return '#8B0000'  # Dark red (high risk, above 5%)
    
    # Create Folium map centered on US
    m = folium.Map(
        location=[39.8283, -98.5795],
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # Create a dictionary for quick lookup
    county_data = county_agg.set_index('fips')[['predicted_cv_change', 'county_name', 'state_name', 'crop']].to_dict('index')
    
    # Calculate min and max for proper binning
    min_val = county_agg['predicted_cv_change'].min()
    max_val = county_agg['predicted_cv_change'].max()
    
    # Add choropleth layer with custom colors
    from branca.colormap import LinearColormap
    
    colormap = LinearColormap(
        colors=['#27ae60', '#FFD700', '#FF8C00', '#8B0000'],
        vmin=min_val,
        vmax=max_val,
        index=[min_val, 0, 2, 5, max_val],
        caption='CV Change (%)'
    )
    
    # Add GeoJson with custom styling
    def style_function(feature):
        fips = feature['id']
        if fips in county_data:
            cv_change = county_data[fips]['predicted_cv_change']
            return {
                'fillColor': get_color(cv_change),
                'fillOpacity': 0.7,
                'color': 'white',
                'weight': 0.3,
                'opacity': 0.3
            }
        # Make counties without data completely invisible
        return {
            'fillOpacity': 0,
            'opacity': 0,
            'weight': 0,
            'color': 'transparent'
        }
    
    def highlight_function(feature):
        return {
            'fillOpacity': 0.8,
            'weight': 1.5,
            'color': 'black'
        }
    
    folium.GeoJson(
        counties_geojson,
        style_function=style_function,
        highlight_function=highlight_function,
    ).add_to(m)
    
    # Add custom tooltips with data
    for feature in counties_geojson['features']:
        fips = feature['id']
        if fips in county_data:
            data = county_data[fips]
            
            # Determine risk level
            if data['predicted_cv_change'] < 0:
                risk_label = "Improving"
                risk_color = "#27ae60"
            elif data['predicted_cv_change'] < 2:
                risk_label = "Low Risk"
                risk_color = "#FFD700"
            elif data['predicted_cv_change'] < 5:
                risk_label = "Medium Risk"
                risk_color = "#FF8C00"
            else:
                risk_label = "High Risk"
                risk_color = "#8B0000"
            
            tooltip_text = f"""
            <div style="font-family: Arial; font-size: 13px; padding: 5px;">
                <b>{data['county_name']}, {data['state_name']}</b><br>
                Crop: {data['crop']}<br>
                CV Change: <b>{data['predicted_cv_change']:.2f}%</b><br>
                Risk Level: <span style="color: {risk_color}; font-weight: bold;">● {risk_label}</span>
            </div>
            """
            
            folium.GeoJson(
                feature,
                style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent', 'weight': 0},
                tooltip=folium.Tooltip(tooltip_text, sticky=True)
            ).add_to(m)
    
    # Add custom legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin: 0 0 10px 0; font-weight: bold;">CV Change Risk Levels</p>
        <p style="margin: 5px 0;"><span style="color: #8B0000; font-size: 20px;">●</span> <b>High Risk</b> (> 5%)</p>
        <p style="margin: 5px 0;"><span style="color: #FF8C00; font-size: 20px;">●</span> <b>Medium Risk</b> (2-5%)</p>
        <p style="margin: 5px 0;"><span style="color: #FFD700; font-size: 20px;">●</span> <b>Low Risk</b> (0-2%)</p>
        <p style="margin: 5px 0;"><span style="color: #27ae60; font-size: 20px;">●</span> <b>Improving</b> (< 0%)</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Display the map - disable returned_objects to prevent reruns on interaction
    st_folium(m, width=1400, height=600, returned_objects=[])
    
    st.info("""

    - **Dark Red**: High risk (CV change > 5%)
    - **Red/Orange**: Medium risk (2-5% increase)
    - **Yellow**: Low risk (0-2% increase)
    - **Green**: Improving (volatility decreasing)
    """)
    
else:
    st.warning("Could not load map data. Showing alternative visualization...")
    
    # Fallback: Scatter plot
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### County Risk Distribution by State")
        
        # Use absolute value for size (can't be negative)
        filtered_data['size_value'] = filtered_data['predicted_cv_change'].abs() + 1
        
        fig = px.scatter(
            filtered_data,
            x='state_name',
            y='predicted_cv_change',
            size='size_value',
            color='risk_level',
            color_discrete_map={
                'High Risk': '#e74c3c',
                'Medium Risk': '#f39c12',
                'Low Risk': '#3498db',
                'Improving': '#27ae60'
            },
            hover_data=['county_name', 'crop', 'yield_cv_change'],
            title="Risk Distribution by State",
            height=500
        )
        
        fig.update_layout(
            xaxis_title="State",
            yaxis_title="Predicted Volatility Change (%)",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Risk Distribution")
        
        # Risk level counts
        risk_counts = filtered_data['risk_level'].value_counts()
        
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={
                'High Risk': '#e74c3c',
                'Medium Risk': '#f39c12',
                'Low Risk': '#3498db',
                'Improving': '#27ae60'
            },
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# State summary
st.markdown("### State-Level Risk Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Top High-Risk States")
    state_risk = filtered_data[filtered_data['predicted_cv_change'] > 5].groupby('state_name').size().sort_values(ascending=False).head(10)
    
    if len(state_risk) > 0:
        fig = px.bar(
            x=state_risk.values,
            y=state_risk.index,
            orientation='h',
            labels={'x': 'Number of High-Risk Counties', 'y': 'State'},
            color=state_risk.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No high-risk counties with current filters")

with col2:
    st.markdown("#### Average Risk by State")
    state_avg = filtered_data.groupby('state_name')['predicted_cv_change'].mean().sort_values(ascending=False).head(10)
    
    fig = px.bar(
        x=state_avg.values,
        y=state_avg.index,
        orientation='h',
        labels={'x': 'Avg Predicted CV Change (%)', 'y': 'State'},
        color=state_avg.values,
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Detailed table
st.markdown("### County-Level Details")

# Add search
search = st.text_input("Search by county or state name")
if search:
    table_data = filtered_data[
        filtered_data['county_name'].str.contains(search, case=False, na=False) |
        filtered_data['state_name'].str.contains(search, case=False, na=False)
    ]
else:
    table_data = filtered_data

# Display table
display_cols = ['county_name', 'state_name', 'crop', 'predicted_cv_change', 
                'yield_cv_change', 'risk_level']

if len(table_data) > 0:
    # Sort by predicted change
    table_data = table_data.sort_values('predicted_cv_change', ascending=False)
    
    st.dataframe(
        table_data[display_cols].head(50),
        use_container_width=True,
        height=400
    )

