#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyodbc
import pandas as pd
from datetime import datetime, timedelta
from holidays import Canada
from dateutil.easter import easter
import numpy as np
from prophet import Prophet

# Establish database connection
dsn_name = "mysnowflake"
connection = pyodbc.connect(f'DSN={dsn_name};')
cursor = connection.cursor()

# Retrieve Snowflake version
cursor.execute("SELECT CURRENT_VERSION ()")
snowflake_version = cursor.fetchone()[0]
print("Snowflake version:", snowflake_version)

# SQL query to retrieve historical data
query = """
    SELECT "Date","Offered" 
    FROM database.livecalls
    WHERE Group = 'English'
    ORDER BY "Date" ASC
"""
df = pd.read_sql(query, connection)
connection.close()

# Data preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['day_of_week_name'] = df['Date'].dt.strftime('%a')
df['Week_Number'] = df['Date'].dt.strftime('%U').astype(int)
df = df.rename(columns={'Offered': 'NCO'})
df['NCO'] = df['NCO'].round().astype(int)
df = df[['Date', 'Year', 'Month', 'Day', 'day_of_week_name', 'NCO']]

# Retrieve Canada holidays
canada_holidays = {}
for year in range(2019, 2028):
    prov_holidays = Canada(years=year, prov='SK').items()
    for date, name in prov_holidays:
        canada_holidays[pd.Timestamp(date)] = name

    easter_sunday = easter(year)
    canada_holidays[easter_sunday] = 'Easter Sunday'
    canada_holidays[easter_sunday + timedelta(days=1)] = 'Easter Monday'

# Create DataFrame for holidays
holiday_df = pd.DataFrame({
    'Holiday': [name for _, name in sorted(canada_holidays.items(), key=lambda x: x[0].date())],
    'Date': sorted([key.date() for key in canada_holidays.keys()])
})
holiday_df['Date'] = pd.to_datetime(holiday_df['Date'])
holidays = holiday_df.copy()
holidays.columns = ['holiday', 'ds']

# Handling outliers
high_weekday_percentiles = df.groupby('day_of_week_name')['NCO'].quantile(0.85)
low_weekday_percentiles25 = df.groupby('day_of_week_name')['NCO'].quantile(0.25)
low_weekday_percentiles10 = df.groupby('day_of_week_name')['NCO'].quantile(0.45)

def adjust_outliers(row):
    percentile_upper = high_weekday_percentiles[row['day_of_week_name']]
    percentile_lower25 = low_weekday_percentiles25[row['day_of_week_name']]
    percentile_lower15 = low_weekday_percentiles10[row['day_of_week_name']]
    if row['NCO'] > percentile_upper:
        return percentile_upper
    elif row['NCO'] < percentile_lower25:
        return percentile_lower15
    else:
        return row['NCO']

df['Adjusted_Daily_NCO'] = df.apply(adjust_outliers, axis=1)

# Applying weighted average
weights = [0.1, 0.2, 0.7]
def weighted_mean(x):
    return sum(x * weights) / sum(weights)
df['Rolling_mean'] = df.groupby('day_of_week_name')['Adjusted_Daily_NCO'].transform(lambda x: x.rolling(3).apply(weighted_mean))
df = df.dropna()

# Forecasting with Prophet
df_new = df[['Date', 'NCO']]
df_new.columns = ['ds', 'y']
m = Prophet(interval_width=0.95, seasonality_mode='multiplicative', daily_seasonality=7, weekly_seasonality=52, yearly_seasonality=12)
model = m.fit(df_new)

# Factoring in holidays
m_holidays = Prophet(holidays=holidays)
model_holidays = m_holidays.fit(df_new)

# Predicting future data
future = m.make_future_dataframe(periods=120, freq='D')
forecast = m.predict(future)
df_forecasted = forecast[['ds', 'yhat']]
df_forecasted.columns = ['Date', 'Forecast_NCO']

# Handling holiday adjustments
holiday_adjustment = { ... }  # Define adjustment factors for holidays
df_forecasted['Holiday'] = forecast['ds'].apply(lambda x: holiday_adjustment.get(x, 1))
df_forecasted['New_forecast'] = df_forecasted['Forecast_NCO'] * df_forecasted['Holiday']

# Additional data analysis and export
df_forecasted['Day_Month'] = df_forecasted['Date'].dt.strftime('%d-%b')
date_adjustment = { ... }  # Define adjustment factors for specific dates in December
df_forecasted['New_forecast2'] = df_forecasted.apply(lambda row: row['New_forecast'] * (1 + date_adjustment.get(row['Day_Month'], 0)), axis=1)

# Calculating daily variance and accuracy
Forecast_Lob_Upload['Daily_Variance'] = abs(Forecast_Lob_Upload['Actual']-Forecast_Lob_Upload['Forecast'])
Forecast_Lob_Upload['Daily_Accuracy'] = abs(100 - (np.where(Forecast_Lob_Upload['Actual'] != 0,(Forecast_Lob_Upload['Daily_Variance']/Forecast_Lob_Upload['Actual'])*100, 0)))

# Calculating weekly actual, forecast, variance and accuracy
Forecast_Lob_Upload['Weekly_Actual'] = Forecast_Lob_Upload.groupby(['Year','Week_Number'])['Actual'].transform('sum')
Forecast_Lob_Upload['Weekly_Forecast'] = Forecast_Lob_Upload.groupby(['Year','Week_Number'])['Forecast'].transform('sum')
Forecast_Lob_Upload['Weekly_Variance'] = abs(Forecast_Lob_Upload['Weekly_Actual']-Forecast_Lob_Upload['Weekly_Forecast'])
Forecast_Lob_Upload['Weekly_Accuracy'] = abs(100 - (np.where(Forecast_Lob_Upload['Weekly_Actual'] != 0,(Forecast_Lob_Upload['Weekly_Variance']/Forecast_Lob_Upload['Weekly_Actual'])*100, 0)))


# Rounding daily and weekly accuracy to 2 decimal places
Forecast_Lob_Upload['Daily_Accuracy'] = round(Forecast_Lob_Upload['Daily_Accuracy'], 2)
Forecast_Lob_Upload['Weekly_Accuracy'] = round(Forecast_Lob_Upload['Weekly_Accuracy'], 2)

# Calculating month
Forecast_Lob_Upload['Month'] = Forecast_Lob_Upload['Date'].dt.month

# Calculating monthly actual, forecast, variance and accuracy
Forecast_Lob_Upload['Monthly_Actual'] = Forecast_Lob_Upload.groupby(['Year','Month'])['Actual'].transform('sum')
Forecast_Lob_Upload['Monthly_Forecast'] = Forecast_Lob_Upload.groupby(['Year','Month'])['Forecast'].transform('sum')
Forecast_Lob_Upload['Monthly_Variance'] = abs(Forecast_Lob_Upload['Monthly_Actual']-Forecast_Lob_Upload['Monthly_Forecast'])
Forecast_Lob_Upload['Monthly_Accuracy'] = abs(100 - (np.where(Forecast_Lob_Upload['Monthly_Actual'] != 0,(Forecast_Lob_Upload['Monthly_Variance']/Forecast_Lob_Upload['Monthly_Actual'])*100, 0)))


# Round accuracy to 2 decimal places
Forecast_Lob_Upload['Monthly_Accuracy'] = round(Forecast_Lob_Upload['Monthly_Accuracy'], 2)

# Finally, save the forecast results to a CSV file
filename = r'path_to_save\forecast_results.csv'
df_forecasted.to_csv(filename, index=False)
print("Updated Successfully")
