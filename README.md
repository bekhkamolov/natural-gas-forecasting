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
