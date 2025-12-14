# Crop Yield Volatility Risk Assessment Dashboard

## Overview

Interactive dashboard analyzing crop yield volatility across US agricultural counties from 2005-2023. Shows how climate change affects corn and soybean production stability.

## Key Findings

- Temperature variability (not just warming) is the primary driver of increased yield volatility
- 97 counties identified as high-risk with significant volatility increases
- Geographic patterns show impacts concentrated in marginal agricultural regions (Great Plains, Southern states)
- Most counties show stable or improving trends, but vulnerable regions are experiencing deterioration

## Results

### Model Performance
- XGBoost selected as best model with R² = 0.566 (explains 56.6% of volatility variation)
- RMSE: 5.95% and MAE: 3.93% on test set
- Cross-validation R² = 0.636 shows good generalization

### Dataset Coverage
- 56,474 observations
- 196 agricultural counties
- 19 years (2005-2023)
- 2 crops (corn and soybean)

### Risk Distribution
- High Risk (>5% CV increase): Counties with extreme volatility increases
- Medium Risk (2-5% CV increase): Moderate volatility increases
- Low Risk (0-2% CV change): Relatively stable counties
- Improving (<0% CV change): Counties with decreasing volatility

## Quick Start

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

## Notebooks Folder

The `notebooks/` folder contains Jupyter notebooks documenting data preprocessing, feature engineering, and model training. These notebooks are not required to run the Streamlit dashboard.

