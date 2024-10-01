**Forecasting with Prophet README**
This repository contains a Python script for forecasting call volumes using Facebook's Prophet library, utilizing historical call data stored in a Snowflake database.
The script performs data preprocessing, outlier handling, holiday adjustment, and forecasting to generate accurate predictions for future call volumes.

**Prerequisites**
Python 3.x
Required Python libraries: pyodbc, pandas, holidays, numpy, prophet

**Setup**
Install the required Python libraries using pip:

pip install pyodbc pandas holidays numpy prophet
Ensure access to a Snowflake database with historical call data.

**Usage**
Update the Snowflake DSN (Data Source Name) in the script with your Snowflake configuration:

dsn_name = "mysnowflake"
Run the script:

python forecast_calls.py
After execution, the forecast results will be saved to a CSV file named forecast_results.csv.

**Script Overview**
Establish Database Connection: Connect to the Snowflake database to retrieve historical call data.

Retrieve Historical Data: Fetch historical call data from the database and preprocess it for analysis.

Retrieve Canada Holidays: Retrieve Canadian holidays including provincial holidays and Easter Sunday/Monday for the forecast period.

Data Preprocessing: Perform data preprocessing tasks such as converting data types, adding date-related features, and renaming columns.

Handling Outliers: Adjust outliers in the call volume data based on weekday percentiles.

Applying Weighted Average: Calculate a weighted rolling mean of call volumes to smooth out fluctuations.

Forecasting with Prophet: Use Facebook's Prophet library to forecast future call volumes.

Factoring in Holidays: Incorporate holiday effects into the forecast model.

Predicting Future Data: Generate predictions for future call volumes.

Handling Holiday Adjustments: Adjust forecasted call volumes based on holiday factors.

Additional Data Analysis: Calculate daily, weekly, and monthly variances and accuracies of the forecasts.

Export Forecast Results: Save the forecast results to a CSV file for further analysis.


**Customization**
Adjust the forecast period and frequency by modifying the periods and freq parameters in the make_future_dataframe function.

Define specific adjustments for holidays and dates in the script as needed.


**Note**
Ensure that the Snowflake database connection details are correctly configured before running the script.

Review and customize the holiday adjustment factors and date adjustments according to your specific requirements.
