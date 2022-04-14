#%% import module

import pandas as pd
import sqlserverconnection as sc
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas.tseries.offsets import MonthEnd

#%%

str_gdp_event_name = 'GDP'
str_leading_indicator = 'ISM Manufacturing PMI'
str_leading_indicator = 'Michigan Consumer Sentiment'
str_currency = 'USD'
str_starting_date = '1/1/2008'
int_future_period = 12

#%%
print("""sudo /opt/mssql/bin/sqlservr""")


#%%
str_sql_query_economic_indicator_gdp = """

SELECT Distinct [Date]
      
      ,[Cur]
      ,[Event]
      ,[Actual]
      ,[Reporting Frequency]
    
  FROM [db_economic_data_indicators].[dbo].[tbl_economic_data_indicator_investing_com]
  WHERE 
    Cur = '{str_currency}'
    and Event = '{str_gdp_event_name}'
    and Actual is not null
    and Date >= '{str_starting_date}'
Order by Date

""".format(str_currency = str_currency, 
           str_gdp_event_name = str_gdp_event_name,
           str_starting_date = str_starting_date)

obj_sql_connection  = sc.CONNECT_TO_SQL_SERVER(_str_server = "LAPTOP-9O71KA1L",
                                            _str_database = 'db_economic_data_indicators',
                                            _str_trusted_connection = 'yes',
                                            str_download_or_upload = 'download')

df_economic_indicator_gdp = pd.read_sql(sql = str_sql_query_economic_indicator_gdp,
                                        con = obj_sql_connection
                                        )

#%%
df_economic_indicator_gdp[str_gdp_event_name] = df_economic_indicator_gdp['Actual']

#%%
str_sql_query_economic_indicator_ism_manufacturing = """

SELECT Distinct [Date]
      
      ,[Cur]
      ,[Event]
      ,[Actual]
      ,[Reporting Frequency]
    
  FROM [db_economic_data_indicators].[dbo].[tbl_economic_data_indicator_investing_com]
  WHERE 
    Cur = '{str_currency}'
    and Event = '{str_leading_indicator}'
    and Actual is not null
    and Date >= '{str_starting_date}'
Order by Date

""".format(str_currency = str_currency, 
          str_leading_indicator = str_leading_indicator,
          str_starting_date =str_starting_date)

df_economic_indicator_leading_indicator = pd.read_sql(sql = str_sql_query_economic_indicator_ism_manufacturing,
                                        con = obj_sql_connection
                                        )
                                        


#%% Create a new column with the currency and leading indicator

str_column_name_currency_with_leading_indicator = f'{str_currency}-{str_leading_indicator}'
df_economic_indicator_leading_indicator[str_column_name_currency_with_leading_indicator] = df_economic_indicator_leading_indicator['Actual']

#%% Create a new column showing the percentage change in the leading indicator

str_column_name_currency_with_leading_indicator_pct_change = f'{str_column_name_currency_with_leading_indicator} Change %'
df_economic_indicator_leading_indicator[str_column_name_currency_with_leading_indicator_pct_change] = df_economic_indicator_leading_indicator[str_column_name_currency_with_leading_indicator].pct_change(1).fillna(0)


#%% Create a new column showing the binary direction of change of the leading indicator. Positive 1 for increase and -1 for decrease

str_column_name_currency_with_leading_indicator_binary_direction = f'{str_column_name_currency_with_leading_indicator} Direction'

df_economic_indicator_leading_indicator[str_column_name_currency_with_leading_indicator_binary_direction] = np.where(df_economic_indicator_leading_indicator[str_column_name_currency_with_leading_indicator_pct_change] >= 0,
                                                                    1,
                                                                    -1)


#%%          

df_combined_gdp_and_leading_indicator = pd.merge(df_economic_indicator_gdp,
                                                 df_economic_indicator_leading_indicator,
                                                 how = 'outer',
                                                 on = 'Date')

#%% Select important columns to keep

list_str_columns_to_keep = ['Date',
                            str_gdp_event_name,
                            str_column_name_currency_with_leading_indicator,
                            str_column_name_currency_with_leading_indicator_pct_change,
                            str_column_name_currency_with_leading_indicator_binary_direction]

df_combined_gdp_and_leading_indicator = df_combined_gdp_and_leading_indicator[list_str_columns_to_keep]


#%% Sort Values by Date

df_combined_gdp_and_leading_indicator = df_combined_gdp_and_leading_indicator.sort_values('Date')

# %% Forward fill both GDP and Leading indicators
list_str_columns = [str_gdp_event_name,
                    str_column_name_currency_with_leading_indicator,
                    str_column_name_currency_with_leading_indicator_pct_change,
                    str_column_name_currency_with_leading_indicator_binary_direction]

df_combined_gdp_and_leading_indicator[list_str_columns] = df_combined_gdp_and_leading_indicator[list_str_columns].fillna(method='ffill')

#%% Remove duplicate ISM and only keep the first record
df_combined_gdp_and_leading_indicator['Date'] = pd.to_datetime(df_combined_gdp_and_leading_indicator['Date'])

#%%
df_combined_gdp_and_leading_indicator['StartOfMonth'] = df_combined_gdp_and_leading_indicator['Date'] + MonthEnd(1)

#%%

df_combined_gdp_and_leading_indicator = df_combined_gdp_and_leading_indicator.drop_duplicates(subset=['StartOfMonth',str_column_name_currency_with_leading_indicator], keep= 'first')

#%%
df_combined_gdp_and_leading_indicator = df_combined_gdp_and_leading_indicator.drop(labels = ['StartOfMonth'], axis = 1)

# %% Get the future GDP to match the present leading indicator

str_column_name_future_gdp = 'Future - ' + str_gdp_event_name
df_combined_gdp_and_leading_indicator[str_column_name_future_gdp] = df_combined_gdp_and_leading_indicator[str_gdp_event_name].shift(-int_future_period)



#%% Create a new column showing the absolute change by getting the difference of the current GDP and the previous GDP

# %% Get the future GDP to match the present leading indicator

str_column_name_future_gdp_absolute_change = 'Future - ' + str_gdp_event_name + 'Aboslute Change'
df_combined_gdp_and_leading_indicator[str_column_name_future_gdp_absolute_change] = df_combined_gdp_and_leading_indicator[str_column_name_future_gdp] - df_combined_gdp_and_leading_indicator[str_column_name_future_gdp].shift(1)

df_combined_gdp_and_leading_indicator[str_column_name_future_gdp_absolute_change]  = df_combined_gdp_and_leading_indicator[str_column_name_future_gdp_absolute_change].fillna(0)

#%% Remove data points that does not have future GDP in it

boolean_filter_condition = df_combined_gdp_and_leading_indicator[str_column_name_future_gdp].notna()

df_combined_gdp_and_leading_indicator = df_combined_gdp_and_leading_indicator[boolean_filter_condition]

#%% Create a new dataframe where the dependent and independent varaiable will be derived from 

df_machine_learning_dataset = df_combined_gdp_and_leading_indicator.drop(labels=['GDP'],axis= 1 )




#%% Set Date as the index
df_machine_learning_dataset = df_machine_learning_dataset.set_index('Date')





# %% Split the dataset and make sure to DO NOT SHUFFLE the data since you are dealing with time series data set

x = df_machine_learning_dataset.drop(labels=[str_column_name_future_gdp, str_column_name_future_gdp_absolute_change], axis = 1)


y = df_machine_learning_dataset[str_column_name_future_gdp]
#y = df_machine_learning_dataset[str_column_name_future_gdp_absolute_change]


#%% 

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.80, shuffle= False)



#%%

model = rf(n_estimators = 1000, max_depth = 1000)

model.fit(X = x_train, y = y_train)


prediction = model.predict(X = x_test)

mae = mean_absolute_error(y_test.to_list(), prediction)
print(f'mae = {mae}')

mse = mean_squared_error(y_test.to_list(), prediction)
print(f'mse = {mse}')

df_prediction = pd.DataFrame(y_test.copy())
df_prediction['Prediction'] = prediction




#%%

import plotly.express as px
fig = px.scatter(df_prediction, x=df_prediction.columns[0], y="Prediction")
fig.show()













#%%