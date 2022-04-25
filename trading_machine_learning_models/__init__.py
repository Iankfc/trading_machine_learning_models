#%% import module


import pandas as pd
import sqlserverconnection as sc
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, accuracy_score
from pandas.tseries.offsets import MonthEnd
from matplotlib import pyplot as plt
from machine_learning_models import model_01_classifier_random_forest_classifier as model01 
from machine_learning_models import model_02_classifier_neural_net as model02
from machine_learning_models import model_03_classifier_gradient_boosting_classifier as model03
from machine_learning_models import model_04_classifier_support_vector_machine as model04
from machine_learning_models import model_05_classifier_quadratic_discriminant as model05
from machine_learning_models import model_06_classifier_linear_discriminant as model06
from machine_learning_models import model_07_classifier_ridge as model07
from machine_learning_models import model_08_classifier_sgd as model08
from machine_learning_models import model_09_classifier_k_nearest_neighbor as model09

#%%

str_gdp_event_name = 'GDP'
str_prediction_currency = 'USD'


tuple_str_currency = ('USD','CAD','MXN','EUR','GBP','CNY','AUD')
str_starting_date = '1/1/2008'
int_future_period = 3
int_random_forest_number_of_trees_estimator = 10_000
int_random_forest_max_dept = 100
float_train_size = 0.80

int_model_code = 6

tuple_hidden_layer = (5,2)
                   

#%%
print("""sudo /opt/mssql/bin/sqlservr""")

obj_sql_connection  = sc.CONNECT_TO_SQL_SERVER(_str_server = "LAPTOP-9O71KA1L",
                                            _str_database = 'db_economic_data_indicators',
                                            _str_trusted_connection = 'yes',
                                            str_download_or_upload = 'download')

#%%

def func_df_leading_indicators():
  
  str_query = """
              SELECT Distinct 
              --Top 10 
              Cur,
              Event
              FROM [db_economic_data_indicators].[dbo].[tbl_economic_data_indicator_investing_com]
              where cur in {tuple_str_currency}
              and Event != '{str_gdp_event_name}'
              """.format(str_gdp_event_name = str_gdp_event_name,
                         tuple_str_currency = tuple_str_currency)

  df_leading_indicators = pd.read_sql(sql = str_query,
                                          con = obj_sql_connection
                                      )
                       
                       
  df_leading_indicators = df_leading_indicators[['Cur','Event']]
  
  bool_condition = df_leading_indicators['Event'].apply(lambda x: "'" in x)
  
  df_leading_indicators = df_leading_indicators[~bool_condition]
  
  return df_leading_indicators 
  
#%%

df_leading_indicators = func_df_leading_indicators()


#%%        
                       
def func_df_economic_indicator_gdp(str_prediction_currency = None,
                                   str_gdp_event_name = None,
                                   str_starting_date = None):
  #%%
  str_sql_query_economic_indicator_gdp = """

  SELECT Distinct [Date]
        
        ,[Cur]
        ,[Event]
        ,[Actual]
        ,[Reporting Frequency]
      
    FROM [db_economic_data_indicators].[dbo].[tbl_economic_data_indicator_investing_com]
    WHERE 
      Cur = '{str_prediction_currency}'
      and Event = '{str_gdp_event_name}'
      and Actual is not null
      and Date >= '{str_starting_date}'
  Order by Date

  """.format(str_prediction_currency = str_prediction_currency, 
            str_gdp_event_name = str_gdp_event_name,
            str_starting_date = str_starting_date)



  df_economic_indicator_gdp = pd.read_sql(sql = str_sql_query_economic_indicator_gdp,
                                          con = obj_sql_connection
                                          )

  #%%
  df_economic_indicator_gdp[str_gdp_event_name] = df_economic_indicator_gdp['Actual']
  
    #%% Convert quarterly gdp into a rolling a
  df_economic_indicator_gdp['TempMonthlyReturn'] = df_economic_indicator_gdp[str_gdp_event_name].apply(lambda x: (1+x)**(1/3)-1) 
  df_economic_indicator_gdp[str_gdp_event_name] = df_economic_indicator_gdp['TempMonthlyReturn'].rolling(int_future_period).apply(lambda x: np.prod(1+ x)-1) 
  
  df_economic_indicator_gdp = df_economic_indicator_gdp.drop(labels=['TempMonthlyReturn'], axis = 1)
    
  df_economic_indicator_gdp[str_gdp_event_name] = np.where(df_economic_indicator_gdp[str_gdp_event_name] > df_economic_indicator_gdp[str_gdp_event_name].shift(1),
                                                           1,
                                                           0)
  
  
  return df_economic_indicator_gdp

#%%


df_economic_indicator_gdp = func_df_economic_indicator_gdp(str_prediction_currency = str_prediction_currency,
                                          str_gdp_event_name = str_gdp_event_name,
                                          str_starting_date = str_starting_date)

  


#%%

def func_df_economic_indicator_leading_indicator(str_currency = None,
                                                 str_leading_indicator = None,
                                                 str_starting_date = None):

  str_sql_query_economic_indicator_leading_indicator = """

  SELECT Distinct 
        [Date]
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
      and [Reporting Frequency] = ''
  Order by Date

  """.format(str_currency = str_currency, 
            str_leading_indicator = str_leading_indicator,
            str_starting_date =str_starting_date)

  df_economic_indicator_leading_indicator = pd.read_sql(sql = str_sql_query_economic_indicator_leading_indicator,
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

  #%% Select important columns to keep

  list_str_columns_to_keep = ['Date',
                            str_column_name_currency_with_leading_indicator,
                            str_column_name_currency_with_leading_indicator_pct_change,
                            str_column_name_currency_with_leading_indicator_binary_direction]

  df_economic_indicator_leading_indicator = df_economic_indicator_leading_indicator[list_str_columns_to_keep]

  return df_economic_indicator_leading_indicator


#%%


df_economic_indicator_leading_indicator = pd.DataFrame([])
for str_currency, str_leading_indicator in df_leading_indicators.itertuples(index = False):

  print(str_currency)
  print(str_leading_indicator)

  df_temp = func_df_economic_indicator_leading_indicator(str_currency = str_currency,
                                                          str_leading_indicator = str_leading_indicator,
                                                          str_starting_date = str_starting_date)

  try:
    df_economic_indicator_leading_indicator = pd.merge(df_economic_indicator_leading_indicator,
                                                      df_temp,
                                                      how = 'outer',
                                                      on = 'Date')
  except KeyError:
    df_economic_indicator_leading_indicator = df_temp.copy()


#%%          

def func_feature_engineering_df_machine_learning_dataset():
  
  

  # %% Get the future GDP to match the present leading indicator

  str_column_name_future_gdp = 'Future - ' + str_gdp_event_name
  df_economic_indicator_gdp[str_column_name_future_gdp] = df_economic_indicator_gdp[str_gdp_event_name].shift(-int_future_period)

    
  
  
  #%%
  df_combined_gdp_and_leading_indicator = pd.merge(df_economic_indicator_gdp,
                                                  df_economic_indicator_leading_indicator,
                                                  how = 'outer',
                                                  on = 'Date')


  #%%

  list_str_column_names_to_be_removed  = ['Cur','Event','Actual','Reporting Frequency']

  df_combined_gdp_and_leading_indicator = df_combined_gdp_and_leading_indicator.drop(labels=list_str_column_names_to_be_removed, 
                                                                                    axis = 1)


  #%% Sort Values by Date

  df_combined_gdp_and_leading_indicator['Date'] = pd.to_datetime(df_combined_gdp_and_leading_indicator['Date'])

  df_combined_gdp_and_leading_indicator = df_combined_gdp_and_leading_indicator.sort_values('Date')

  df_combined_gdp_and_leading_indicator = df_combined_gdp_and_leading_indicator.set_index('Date')

  # %% Forward fill both GDP and Leading indicators

  df_combined_gdp_and_leading_indicator = df_combined_gdp_and_leading_indicator.fillna(method='ffill')


  #%% Remove GDP

  df_combined_gdp_and_leading_indicator = df_combined_gdp_and_leading_indicator.drop(labels = [str_gdp_event_name], axis = 1)


  #%% Remove nas under future gdp

  bool_condition = df_combined_gdp_and_leading_indicator[str_column_name_future_gdp].notna()

  df_combined_gdp_and_leading_indicator = df_combined_gdp_and_leading_indicator[bool_condition]
  
  df_combined_gdp_and_leading_indicator = df_combined_gdp_and_leading_indicator.fillna(0)
  
  df_combined_gdp_and_leading_indicator = df_combined_gdp_and_leading_indicator.replace([np.inf, -np.inf], 
                                                                                        0)

  x = df_combined_gdp_and_leading_indicator.drop(labels=[str_column_name_future_gdp, str_column_name_future_gdp], axis = 1)

  y = df_combined_gdp_and_leading_indicator[str_column_name_future_gdp]

  return x, y

#%%


def function_df_train_test_split():

  x, y = func_feature_engineering_df_machine_learning_dataset()


  # %% Split the dataset and make sure to DO NOT SHUFFLE the data since you are dealing with time series data set

  x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                      train_size = float_train_size,
                                                      shuffle= False)



  return x_train, x_test, y_train, y_test


#%%

x_train, x_test, y_train, y_test = function_df_train_test_split()


#%%


def funct_dict_model_catalog(int_model_code = None):
  dict_model_catalog = {
                        1: model01.func_run_model(x_train = x_train,
                                                  x_test = x_test,
                                                  y_train = y_train,
                                                  y_test = y_test,
                                                  int_random_forest_number_of_trees_estimator = int_random_forest_number_of_trees_estimator,
                                                  int_random_forest_max_dept = int_random_forest_max_dept),
  
                        2: model02.func_run_model(x_train = x_train,
                                                  x_test = x_test,
                                                  y_train = y_train,
                                                  y_test = y_test,
                                                  tuple_hidden_layer = tuple_hidden_layer),
  
                        3: model03.func_run_model(x_train = x_train,
                                                  x_test = x_test,
                                                  y_train = y_train,
                                                  y_test = y_test),
                        
                        4: model04.func_run_model(x_train = x_train,
                                                  x_test = x_test,
                                                  y_train = y_train,
                                                  y_test = y_test),
                        
                        5: model05.func_run_model(x_train = x_train,
                                                  x_test = x_test,
                                                  y_train = y_train,
                                                  y_test = y_test),
                        
                        6: model06.func_run_model(x_train = x_train,
                                                  x_test = x_test,
                                                  y_train = y_train,
                                                  y_test = y_test),
                        
                        7: model07.func_run_model(x_train = x_train,
                                                  x_test = x_test,
                                                  y_train = y_train,
                                                  y_test = y_test),
                        
                        8: model08.func_run_model(x_train = x_train,
                                                  x_test = x_test,
                                                  y_train = y_train,
                                                  y_test = y_test),
                        
                        9: model09.func_run_model(x_train = x_train,
                                                  x_test = x_test,
                                                  y_train = y_train,
                                                  y_test = y_test),
                        }


  return dict_model_catalog[int_model_code]
  


#%%


dict_model_results = funct_dict_model_catalog(int_model_code = int_model_code)
df_prediction = dict_model_results['df_prediction']
df_feature_importance =  dict_model_results['df_feature_importance']
accuracy_score =  dict_model_results['float_accuracy_score']

#%%



