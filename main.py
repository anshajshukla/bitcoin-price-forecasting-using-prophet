"""
# Bitcoin Price Forecast with Scenarios (2024 - 2028) using Prophet.

## Author: Iman Samizadeh
## Contact: Iman.samizadeh@gmail.com
## License: MIT License (See below)

MIT License

Copyright (c) 2024 Iman Samizadeh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE
DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY,
WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Disclaimer

This code and its predictions are for educational purposes only and should not be considered as financial or investment advice.
The author and anyone associated with the code is not responsible for any financial losses or decisions based on the code's output.
"""

import pandas as pd
from prophet import Prophet
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ta
from data_helper import DataHelper

halving_dates = [datetime(2012, 11, 28), datetime(2016, 7, 9), datetime(2020, 5, 11), datetime(2024, 5, 12), datetime(2028, 5, 12)]
holidays_df = pd.DataFrame({
    'ds': halving_dates,
    'holiday': 'halving',
})

# Fetching historical data
data = DataHelper('btcusd', 'd1')
btc_data = data.fetch_historical_data()
btc_data['ds'] = pd.to_datetime(btc_data['timestamp'], unit='ms')
btc_data['y'] = btc_data['close']
btc_data['volume'] = btc_data['volume']

model = Prophet(daily_seasonality=True)
model.add_regressor('volume')

# Fitting the model
model.fit(btc_data[['ds', 'y', 'volume']])

# Calculating RSI
btc_data['rsi'] = ta.momentum.RSIIndicator(btc_data['y'], window=14).rsi()

# Initializing and fitting the Prophet model with holidays and RSI as a regressor
model = Prophet(daily_seasonality=True, holidays=holidays_df)
btc_data['rsi'].fillna(btc_data['rsi'].mean(), inplace=True)  # Fill missing values in the 'rsi' column with the mean
model.add_regressor('rsi')
model.fit(btc_data[['ds', 'y', 'rsi']])

# Creating a DataFrame for future dates (2024-2028)
future_dates = model.make_future_dataframe(periods=(2028-2024)*365, freq='D')
future_dates = future_dates.merge(btc_data[['ds', 'volume', 'rsi']], on='ds', how='left')
future_dates['volume'] = future_dates['volume'].ffill()  # Forward-fill the missing values
future_dates['rsi'] = future_dates['rsi'].ffill()  # Forward-fill the missing values

# Predicting future prices
forecast = model.predict(future_dates)

# Extracting the volume data for the time when BTC reached its highest price
highest_price_date = btc_data[btc_data['y'] == btc_data['y'].max()]['ds'].iloc[0]
volume_at_highest_price = btc_data[btc_data['ds'] == highest_price_date]['volume'].iloc[0]

# Creating a DataFrame for future dates (2024-2028) with the volume at the highest price
future_dates_2024 = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
forecast_2024 = pd.DataFrame({'ds': future_dates_2024})
forecast_2024 = forecast_2024.merge(btc_data[['ds', 'rsi']], on='ds', how='left')
forecast_2024['rsi'] = forecast_2024['rsi'].ffill()  # Forward-fill the missing values

# Using the Prophet model to predict the price for 2024 with the updated 'rsi' data
forecast_2024 = model.predict(forecast_2024)

plt.style.use('dark_background')
plt.figure(figsize=(20, 10))
plt.plot(btc_data['ds'], btc_data['y'], label='Actual Prices', color='cyan', linewidth=3)
plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Forecasted Prices (Likely)', color='orange', linestyle='--', linewidth=3)
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='grey', alpha=0.5, label='Optimistic & Pessimistic Scenarios')

# Annotating the maximum forecasted price in the likely scenario
max_forecast_likely = forecast.loc[forecast['yhat'].idxmax()]
max_price_likely = max_forecast_likely['yhat']
max_date_likely = max_forecast_likely['ds']
plt.annotate(f'Projected Max (Likely): ${max_price_likely:,.2f}', xy=(max_date_likely, max_price_likely), xytext=(max_date_likely - timedelta(days=365), max_price_likely * 1.1), arrowprops=dict(facecolor='orange', arrowstyle='->'), fontsize=12, color='orange')


# Annotating the maximum forecasted price in the optimistic scenario
max_forecast_optimistic = forecast.loc[forecast['yhat_upper'].idxmax()]
max_price_optimistic = max_forecast_optimistic['yhat_upper']
max_date_optimistic = max_forecast_optimistic['ds']
plt.annotate(f'Projected Max (Optimistic): ${max_price_optimistic:,.2f}',
             xy=(max_date_optimistic, max_price_optimistic),
             xytext=(max_date_optimistic - timedelta(days=365), max_price_optimistic * 1.1),
             arrowprops=dict(facecolor='orange', arrowstyle='->'),
             fontsize=12, color='orange')

def human_friendly_dollar(x, pos):
    if x >= 1e6:
        return '${:1.1f}M'.format(x * 1e-6)
    elif x >= 1e3:
        return '${:1.0f}K'.format(x * 1e-3)
    return '${:1.0f}'.format(x)


# Annotating key Bitcoin halving dates
for i, halving_date in enumerate(halving_dates):
    plt.axvline(x=halving_date, color='magenta', linestyle=':', linewidth=2)
    plt.text(halving_date, plt.ylim()[1] - i * max_price_likely * 0.1, f'Halving {halving_date.year}',
             horizontalalignment='right', color='magenta')

# Annotating the current price
current_price = btc_data['y'].iloc[-1]
current_date = btc_data['ds'].iloc[-1]
plt.annotate(f'Current Price: ${current_price:,.2f}',
             xy=(current_date, current_price),
             xytext=(current_date - timedelta(days=180), current_price * 1.1),
             arrowprops=dict(facecolor='cyan', arrowstyle='->'),
             fontsize=12, color='cyan')

# Plotting the 2024 forecasted price
plt.plot(forecast_2024['ds'], forecast_2024['yhat'], label='2024 Forecast', color='green', linestyle='--', linewidth=3)

# Finding and setting the maximum price in the forecasted data for 2024
max_price_2024 = forecast_2024['yhat'].max()
forecast_2024['yhat'] = max_price_2024
projected_price_2024 = forecast_2024['yhat'].iloc[-1]
projected_date_2024 = forecast_2024['ds'].iloc[-1]

# Annotating the projected price for 2024 (highest)
plt.annotate(f'Projected Price in 2024 (Highest): ${max_price_2024:,.2f}',
             xy=(projected_date_2024, max_price_2024),
             xytext=(projected_date_2024 + timedelta(days=180), max_price_2024 * 1.1),
             arrowprops=dict(facecolor='green', arrowstyle='->'),
             fontsize=12, color='green')

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(human_friendly_dollar))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()
plt.title('Bitcoin Price Forecast with Scenarios (2024 - 2028) using Prophet', fontsize=24, color='white')
plt.xlabel('Date', fontsize=18, color='white')
plt.ylabel('BTC Price (USD)', fontsize=18, color='white')
plt.legend(loc='upper left', fontsize=14)

plt.show()
