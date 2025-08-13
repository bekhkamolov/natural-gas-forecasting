# Natural Gas Price Forecasting

## Overview
Time series forecasting project using ARIMA/SARIMA models to predict natural gas prices based on fundamental market data including storage levels and weather patterns.

## Technologies Used
- Python
- ARIMA/SARIMA models
- XGBoost for comparison
- EIA API for real market data

## Project Structure
- `utils/`: Python source code
- `notebooks/`: Data analysis and modeling
- `data/`: Raw and processed datasets

## Notebooks
_[data_preprocessing](https://github.com/bekhkamolov/natural-gas-forecasting/blob/main/notebooks/data_preprocessing.ipynb)_ notebook prepares and cleans the raw datasets for further time series modeling (ARIMA, SARIMA, ARIMAX). It:

* Loads Henry Hub price data and weather data (temperature, HDD, CDD) from our created packages
* Flattens multi-index column headers
* Aligns and merges datasets on a full continuous daily date range
* Forward-fills missing price values and interpolates weather variables
* Outputs a gap-free dataset (natural_gas_data.csv) ready for modeling

Example output:

| date       | price | temperature | hdd | cdd |
| ---------- | ----- | ----------- | --- | --- |
| 2020-01-02 | 2.122 | 45.3        | 5.0 | 0.0 |
| 2020-01-03 | 2.130 | 44.8        | 6.2 | 0.0 |

## Exploratory Data Analysis ([EDA](https://github.com/bekhkamolov/natural-gas-forecasting/blob/main/notebooks/exploratory_data_analysis.ipynb))

### Data Overview
- **Henry Hub Natural Gas Prices**: Daily prices from July 2021 to July 2025
- **Temperature Data**: Daily average temperatures with clear seasonal patterns
- **Weather Variables**: Heating Degree Days (HDD) and Cooling Degree Days (CDD)

### Key Findings

**Price Series Analysis:**
- High volatility with price spike reaching ~$9.5 USD in mid-2022
- Non-stationary behavior confirmed by ADF test (p-value = 0.46)
- ACF shows slow decay typical of trending data
- PACF indicates AR(1) or AR(2) structure after differencing

**Stationarity Transformation:**
- Applied first differencing to achieve stationarity
- Post-differencing ADF test: p-value = 2.89e-26 (highly significant)
- Differenced series ready for ARIMA modeling

**Variable Relationships:**
- Weak linear correlation between price and weather variables
- Strong correlations between temperature and degree days (HDD: -0.97, CDD: 0.72)
- Suggests complex non-linear relationships may exist

**Modeling Implications:**
- First-differenced series suitable for ARIMA modeling
- PACF suggests starting with AR(1) or AR(2) models
- Weather variables available for multivariate approaches

## [Model Implementations](https://github.com/bekhkamolov/natural-gas-forecasting/blob/main/notebooks/model_implementations.ipynb)

### Models Tested
- **ARIMA Models**: Manual configuration and random walk model based on EDA insights
- **SARIMA Models**: Automated parameter detection using pmdarima
- **XGBoost**: Time series forecasting with engineered features  
- **Model Comparison**: Performance evaluation across different approaches

### ARIMA Results
- **Model Selection**: ARIMA(0,1,0) - Random walk model
- **Rationale**: EDA showed white noise pattern after first differencing, indicating random walk behavior
- **Implementation**: Applied to stationary differenced price series
- **Performance Metrics**:
  - MSE: 0.0185
  - RMSE: 0.1360  
  - MAE: 0.0881
- **Key Findings**: Random walk model effectively captured the unpredictable nature of gas price changes

### SARIMA Analysis
- **Automated Parameter Detection**: Used `pmdarima.auto_arima()` for optimal seasonal parameter selection
- **Seasonal Decomposition**: Analyzed multiplicative seasonal patterns in original price series
- **Model Selection**: Automatic detection of best SARIMA(p,d,q)(P,D,Q,s) configuration
- **Seasonal Periods**: Algorithm tested different seasonal frequencies automatically
- **Results**: Compared seasonal vs non-seasonal model performance

### Feature Engineering (XGBoost)
- **Lag Features**: Historical price differences as predictors
- **Rolling Statistics**: Moving averages and volatility measures
- **Weather Variables**: Temperature, HDD, and CDD integration
- **Time-based Features**: Seasonal and trend components

### Model Evaluation
- **Train/Test Split**: 80-20 temporal split maintaining time series structure
- **Automated Selection**: Used statistical criteria (AIC/BIC) for model comparison
- **Metrics**: MSE, RMSE, and MAE for comprehensive performance assessment

### Key Insights
- Auto ARIMA efficiently identified optimal parameters
- SARIMA modeling revealed seasonal components in gas price data
- Feature engineering improved XGBoost performance significantly
- Weather variables showed limited linear relationship but potential for non-linear modeling
- Automated model selection reduced overfitting risks