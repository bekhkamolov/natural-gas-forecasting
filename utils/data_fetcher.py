
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import time


class NaturalGasDataFetcher:
    """
    Comprehensive data fetcher for natural gas forecasting project
    """

    def __init__(self, eia_api_key=None):
        """
        Initialize with EIA API key (get free key from eia.gov/opendata/)
        """
        self.eia_api_key = eia_api_key
        self.base_eia_url = "https://api.eia.gov/v2"

    def get_eia_data(self, series_id, start_date=None, end_date=None):
        """
        Fetch data from EIA API

        Parameters:
        - series_id: EIA series identifier
        - start_date: Start date (YYYY-MM-DD)
        - end_date: End date (YYYY-MM-DD)
        """
        if not self.eia_api_key:
            print("Warning: No EIA API key provided. Using demo data...")
            return self._generate_demo_data()

        url = f"{self.base_eia_url}/natural-gas/stor/wkly/data/"
        params = {
            'api_key': self.eia_api_key,
            'frequency': 'weekly',
            'data[0]': 'value',
            'sort[0][column]': 'period',
            'sort[0][direction]': 'desc',
            'offset': 0,
            'length': 5000
        }

        if start_date:
            params['start'] = start_date
        if end_date:
            params['end'] = end_date

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'data' in data:
                df = pd.DataFrame(data['data'])
                df['period'] = pd.to_datetime(df['period'])
                df = df.sort_values('period').reset_index(drop=True)
                return df
            else:
                print("No data found in API response")
                return None

        except Exception as e:
            print(f"Error fetching EIA data: {e}")
            print("Using demo data instead...")
            return self._generate_demo_data()

    def get_henry_hub_prices(self, start_date='2020-01-01', end_date=None):
        """
        Get Henry Hub natural gas futures prices from Yahoo Finance
        """
        try:
            # Henry Hub Natural Gas futures ticker
            ticker = 'NG=F'

            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # Download data
            data = yf.download(ticker, start=start_date, end=end_date)

            # Clean and prepare data
            df = data.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.rename(columns={'Date': 'date', 'Close': 'henry_hub_price'})
            df = df[['date', 'henry_hub_price']].dropna()

            print(f"Downloaded {len(df)} Henry Hub price records")
            return df

        except Exception as e:
            print(f"Error fetching Henry Hub data: {e}")
            return self._generate_demo_price_data()

    def get_storage_data(self):
        """
        Get natural gas storage levels from EIA
        """
        # EIA Natural Gas Underground Storage
        storage_series = "NG.NW2_EPG0_SWO_R48_BCF.W"

        storage_data = self.get_eia_data(
            series_id=storage_series,
            start_date='2020-01-01'
        )

        if storage_data is not None:
            storage_data = storage_data.rename(columns={
                'period': 'date',
                'value': 'storage_level'
            })
            storage_data = storage_data[['date', 'storage_level']]
            print(f"Downloaded {len(storage_data)} storage records")

        return storage_data

    def get_weather_data(self, city="Chicago", days_back=365 * 4):
        """
        Get basic weather data (temperature) for heating degree days calculation
        Note: This is a simplified version. For production, use dedicated weather APIs
        """
        print("Note: Using simplified weather data generation")
        print("For production, consider using OpenWeatherMap API or NOAA data")

        # Generate synthetic weather data with seasonal patterns
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days_back),
            end=datetime.now(),
            freq='D'
        )

        # Simulate temperature with seasonal variation
        day_of_year = dates.dayofyear
        base_temp = 50 + 30 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        noise = np.random.normal(0, 10, len(dates))
        temperature = base_temp + noise

        weather_df = pd.DataFrame({
            'date': dates,
            'temperature': temperature
        })

        # Calculate heating degree days (HDD) and cooling degree days (CDD)
        weather_df['hdd'] = np.maximum(65 - weather_df['temperature'], 0)
        weather_df['cdd'] = np.maximum(weather_df['temperature'] - 65, 0)

        return weather_df

    def combine_all_data(self):
        """
        Fetch and combine all datasets
        """
        print("Fetching all natural gas data...")

        # Get price data
        prices = self.get_henry_hub_prices()
        time.sleep(1)  # Be nice to APIs

        # Get storage data
        storage = self.get_storage_data()
        time.sleep(1)

        # Get weather data
        weather = self.get_weather_data()

        # Merge datasets
        if prices is not None and storage is not None:
            # Convert to weekly data for consistency
            prices['week'] = prices['date'].dt.to_period('W')
            storage['week'] = storage['date'].dt.to_period('W')
            weather['week'] = weather['date'].dt.to_period('W')

            # Aggregate to weekly
            weekly_prices = prices.groupby('week')['henry_hub_price'].mean().reset_index()
            weekly_storage = storage.groupby('week')['storage_level'].last().reset_index()
            weekly_weather = weather.groupby('week')[['hdd', 'cdd']].sum().reset_index()

            # Merge all data
            combined = weekly_prices.merge(weekly_storage, on='week', how='inner')
            combined = combined.merge(weekly_weather, on='week', how='left')

            # Convert week back to date
            combined['date'] = combined['week'].dt.start_time
            combined = combined.drop('week', axis=1)
            combined = combined.sort_values('date').reset_index(drop=True)

            print(f"Combined dataset has {len(combined)} weekly records")
            print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")

            return combined

        return None

    def _generate_demo_data(self):
        """
        Generate demo storage data for testing without API key
        """
        print("Generating demo storage data...")
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='W')

        # Simulate seasonal storage pattern
        day_of_year = dates.dayofyear
        seasonal_pattern = 2000 + 1500 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        noise = np.random.normal(0, 100, len(dates))
        storage_levels = np.maximum(seasonal_pattern + noise, 500)  # Minimum storage

        return pd.DataFrame({
            'date': dates,
            'storage_level': storage_levels
        })

    def _generate_demo_price_data(self):
        """
        Generate demo price data
        """
        print("Generating demo price data...")
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')

        # Simulate price with trend and seasonality
        trend = np.linspace(2.5, 4.0, len(dates))
        seasonal = 0.5 * np.sin(2 * np.pi * dates.dayofyear / 365)
        noise = np.random.normal(0, 0.3, len(dates))
        prices = np.maximum(trend + seasonal + noise, 1.0)  # Minimum price

        return pd.DataFrame({
            'date': dates,
            'henry_hub_price': prices
        })


# Usage example:
if __name__ == "__main__":
    # Initialize fetcher (add your EIA API key)
    fetcher = NaturalGasDataFetcher(eia_api_key="oFFfwEonZjh6oENLFfK0XBHeH7nUUcCb0YJh0LJx")

    # Option 1: Get individual datasets
    # prices = fetcher.get_henry_hub_prices()
    # storage = fetcher.get_storage_data()
    # weather = fetcher.get_weather_data()

    # Option 2: Get combined dataset (recommended)
    data = fetcher.combine_all_data()

    if data is not None:
        print("\nDataset overview:")
        print(data.head())
        print(f"\nColumns: {list(data.columns)}")
        print(f"Shape: {data.shape}")

        # Save to CSV for easy access
        data.to_csv('natural_gas_data.csv', index=False)
        print("\nData saved to 'natural_gas_data.csv'")