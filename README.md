# Crop Yield Volatility Risk Assessment Dashboard

## Overview

Interactive dashboard analyzing crop yield volatility across US agricultural counties from 2005-2023. Shows how climate change affects corn and soybean production stability.


### 1. Install Python
You need Python 3.8 or higher installed on your computer.

### 2. Download and Setup

```bash
# Navigate to project folder
cd crop_risk_dashboard

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Add Data Files

Put these CSV files in the `data/` folder:
- `model_predictions.csv`
- `volatility_final_analysis.csv`
- `merged_crop_climate_data.csv`
- `model_comparison_metrics.csv`
- `feature_importance.csv`

Put this model file in the `models/` folder:
- `xgboost_model.pkl`

### 4. Run the App

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## What Each Page Does

### Home
Summary of project with key statistics and top 10 highest risk counties.

### Risk Map
Interactive map showing which counties have high, medium, or low risk based on historical volatility.

### County Explorer
Detailed view of individual counties with yield trends and climate data.

### Volatility Impact Modeler
Adjust climate parameters to see predicted impact on crop volatility.

### Analytics
Charts showing feature importance, correlations, and risk distributions.

### Model Performance
Comparison of three machine learning models (Linear Regression, Random Forest, XGBoost).

## Common Issues

### App won't start
```bash
pip install --upgrade streamlit
```

### Missing data files
Make sure all CSV files are in the `data/` folder with exact names (case-sensitive).

### Model not working
Check that `xgboost_model.pkl` is in the `models/` folder.

### Port already in use
```bash
streamlit run app.py --server.port 8502
```


## Data Sources

- NASA POWER (climate data)
- MODIS (satellite vegetation data)
- USDA NASS (crop yield data)

