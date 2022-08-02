import trading_machine_learning_models as tm
import pandas as pd
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split

x, y = make_classification(n_samples=100, random_state=1)
x = pd.DataFrame(x)
y = pd.DataFrame(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                train_size = 0.50,
                                                shuffle= False)
#%% You can manually edit the hyper parameters
nparray_n_estimators = list(np.arange(1,10,1, dtype=int))
nparray_max_depth = list(np.arange(1,10,1, dtype=int))

#%% Use this function to get the dictionary format at which you can manually edit the list of hyper parameters
dict_list_class_ml_classifier_models = tm.func_dict_list_class_ml_classifier_models(RandomForestClassifier = {'n_estimators':nparray_n_estimators,
                                                                                                            'max_depth':nparray_max_depth
                                                                                                            })
print(dict_list_class_ml_classifier_models)

df_prediction = tm.func_df_run_all_models( dict_list_class_ml_classifier_models = dict_list_class_ml_classifier_models,
                                        x_train = x_train,
                                        y_train = y_train,
                                        x_test = x_test,
                                        y_test = y_test)

print(df_prediction)