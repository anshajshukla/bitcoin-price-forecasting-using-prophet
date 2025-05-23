import os
import glob

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

class DataHelper:
    def __init__(self, symbol, timeframe, data_dir="data/btcusd"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_dir = data_dir

    def fetch_historical_data(self):
        """Fetch historical data from CSV files or fallback to yfinance"""
        # Try to read from CSV files first
        file_pattern = os.path.join(self.data_dir, f"{self.symbol}-{self.timeframe}-*.csv")
        csv_files = glob.glob(file_pattern)
        
        if csv_files:
            df_list = []
            for csv_file in sorted(csv_files):
                try:
                    df = pd.read_csv(csv_file)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df_list.append(df)
                except Exception as e:
                    print(f"Error processing file {csv_file}: {e}")
            
            if df_list:
                return pd.concat(df_list, ignore_index=True)
        
        # Fallback to yfinance if no CSV files found
        try:
            print("No CSV files found. Fetching data from Yahoo Finance...")
            btc = yf.Ticker("BTC-USD")
            df = btc.history(period="max")
            
            # Convert the data to match expected format
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'timestamp',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Convert timestamp to milliseconds
            df['timestamp'] = df['timestamp'].astype(np.int64) // 10**6
            
            return df[['timestamp', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching data from Yahoo Finance: {e}")
            print("Generating sample data...")
            
            # Generate sample data as last resort
            dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
            sample_data = {
                'timestamp': [int(d.timestamp() * 1000) for d in dates],
                'close': np.linspace(5000, 50000, len(dates)) + np.random.normal(0, 1000, len(dates)),
                'volume': np.random.uniform(1e8, 1e9, len(dates))
            }
            return pd.DataFrame(sample_data)

    

    def get_halving_date(self, year):
        halving_dates = self.halving_dates()
        for date in halving_dates:
            if date.year == year:
                return date
        raise ValueError(f"No halving date found for the year {year}")

    def halving_dates(self):
        return [datetime(2012, 11, 28), datetime(2016, 7, 9), datetime(2020, 5, 11), datetime(2024, 5, 12),
                datetime(2028, 5, 12)]

    def days_since_last_halving(self, current_date):
        halving_dates = self.halving_dates()
        past_halvings = [date for date in halving_dates if date < current_date]
        if not past_halvings:
            return 0
        last_halving = max(past_halvings)
        return (current_date - last_halving).days

    def generate_future_features(self, data, features, days=90):
        # Get the last known value for each feature
        last_values = data[features].iloc[-1]

        # Calculate historical percentage changes
        pct_changes = data['close'].pct_change().dropna()

        # Focus on percentage changes around all-time highs (e.g., within 5% of the max price)
        all_time_high = data['close'].max()
        threshold = all_time_high * 0.05
        at_high_pct_changes = pct_changes[data['close'] > (all_time_high - threshold)]

        # Calculate the average percentage change around all-time highs
        avg_pct_change_at_high = at_high_pct_changes.mean()

        # Create a DataFrame to hold future feature values
        future_data = pd.DataFrame(index=pd.date_range(start=data['timestamp'].iloc[-1] + timedelta(days=1), periods=days))

        # Generate future feature values using the average percentage change
        for feature in features:
            future_values = [last_values[feature]]
            for _ in range(1, days):
                future_values.append(future_values[-1] * (1 + avg_pct_change_at_high))
            future_data[feature] = future_values

        return future_data

    def generate_features_to_halving(self, data, features):
        # Get the last known date and value for each feature
        last_known_date = data['timestamp'].iloc[-1]
        last_values = data[features].iloc[-1]

        # Determine the halving date
        halving_date = self.get_halving_date(2024)  # Replace with actual function to get halving date

        # Calculate the number of days to the halving date
        days_to_halving = (halving_date - last_known_date).days

        # Calculate historical percentage changes
        pct_changes = data['close'].pct_change().dropna()

        # Focus on percentage changes around all-time highs (for example)
        all_time_high = data['close'].max()
        threshold = all_time_high * 0.05
        at_high_pct_changes = pct_changes[data['close'] > (all_time_high - threshold)]

        # Calculate the average percentage change around all-time highs
        avg_pct_change_at_high = at_high_pct_changes.mean()

        # Create a DataFrame to hold future feature values
        future_data = pd.DataFrame(index=pd.date_range(start=last_known_date + timedelta(days=1), periods=days_to_halving))

        # Generate future feature values using the average percentage change
        for feature in features:
            future_values = [last_values[feature]]
            for _ in range(1, days_to_halving):
                future_values.append(future_values[-1] * (1 + avg_pct_change_at_high))
            future_data[feature] = future_values

        return future_data

