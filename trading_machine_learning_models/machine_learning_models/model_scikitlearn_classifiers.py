#%% Import machine learning models from sklearn classifiers

from sklearn.ensemble import RandomForestClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier


#%% Import miscellanous modules
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
try:
    from .module_grid_search import func_dict_grid_search
except ImportError:
    from module_grid_search import func_dict_grid_search
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from datetime import datetime
import json

def func_dict_list_class_ml_classifier_models_format():
    
    nparray_n_estimators = list(np.arange(1,10,1, dtype=int))
    nparray_max_depth = list(np.arange(1,10,1, dtype=int))

    dict_list_class_ml_classifier_models = { 0: [RandomForestClassifier, 
                                            {'n_estimators':nparray_n_estimators,
                                            'max_depth':nparray_max_depth
                                            }
                                            ],
                                            1: [MLPClassifier,
                                                        {}],
                                            2: [GradientBoostingClassifier,
                                                        {}],
                                            3: [SVC,
                                                        {}],
                                            4: [LinearDiscriminantAnalysis,
                                                        {}],
                                            5: [RidgeClassifier,
                                                        {}],
                                            6: [SGDClassifier,
                                                        {}],
                                            7: [KNeighborsClassifier,
                                                        {}],
                                            8: [GaussianProcessClassifier,
                                                        {}],
                                            9: [GaussianNB,
                                                        {}],
                                            10: [BernoulliNB,
                                                        {}],
                                            11: [DecisionTreeClassifier,
                                                        {}],
                                            12: [ExtraTreesClassifier,
                                                        {}],
                                            13: [AdaBoostClassifier,
                                                        {}]
                                            }
    return dict_list_class_ml_classifier_models


#%%
def func_run_model(x_train = None,
                    x_test = None,
                    y_train = None,
                    y_test = None,
                    bool_include_x_train_independent_variables = True,
                    class_machine_learning_model = None,
                    **kwargs ):
    
    
    if kwargs == None:
        kwargs = {}
    
    str_model_name = class_machine_learning_model.__name__
    dict_best_hyperparameters = func_dict_grid_search(x_train = x_train,
                                                    y_train = y_train,
                                                    class_machine_learning_model = class_machine_learning_model(),
                                                    dict_nparray_parameter_grid = kwargs)


    model = class_machine_learning_model(**dict_best_hyperparameters)
    model.fit(X = x_train, y = y_train)
    prediction = model.predict(X = x_test)
    df_prediction = pd.DataFrame(y_test.copy())
    df_prediction = df_prediction.rename(columns = {0:'Actual'})
    df_prediction['Prediction'] = prediction
    df_prediction['Prediction'] = df_prediction['Prediction'].astype(int)
    if bool_include_x_train_independent_variables:
        df_prediction = pd.merge(df_prediction, x_test, how = 'left', left_index= True, right_index= True)
    float_model_classification_accuracy = accuracy_score(y_true= y_test, y_pred=prediction)
    
    df_prediction['Accuracy']  = float_model_classification_accuracy
    df_prediction['Accuracy'] = df_prediction['Accuracy'].astype(int)
    df_prediction['ModelName'] = str_model_name
    
    df_prediction['GridSearchParameters'] = np.array(kwargs, dtype = object)
    df_prediction['GridSearchBestParameters'] = np.array(dict_best_hyperparameters, dtype = object)
    
    
    print(f"Model: {str_model_name}")  
    print(f"Accuracy: {float_model_classification_accuracy * 100}%")
    print(f"Confusion Matrix: \n {confusion_matrix(y_true = y_test, y_pred = prediction)}")
    print(f"Best Hyper Parameters: \n {dict_best_hyperparameters}")
    print(f"Hyper Parameters Grid: \n {kwargs}")
    
    
    
    return df_prediction


#%%

if __name__ == '__main__':
    x, y = make_classification(n_samples=100, random_state=1)
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                    train_size = 0.50,
                                                    shuffle= False)
    
    
    
    
#%%
#%% Use this function to get the dictionary format at which you can manually edit the list of hyper parameters
dict_list_class_ml_classifier_models = func_dict_list_class_ml_classifier_models_format()

#%% You can manually edit the hyper parameters
nparray_n_estimators = list(np.arange(1,10,1, dtype=int))
nparray_max_depth = list(np.arange(1,10,1, dtype=int))

dict_list_class_ml_classifier_models = { 0: [RandomForestClassifier, 
                                        {'n_estimators':nparray_n_estimators,
                                        'max_depth':nparray_max_depth
                                        }
                                        ],
                                        1: [MLPClassifier,
                                                    {}],
                                        2: [GradientBoostingClassifier,
                                                    {}],
                                        3: [SVC,
                                                    {}],
                                        4: [LinearDiscriminantAnalysis,
                                                    {}],
                                        5: [RidgeClassifier,
                                                    {}],
                                        6: [SGDClassifier,
                                                    {}],
                                        7: [KNeighborsClassifier,
                                                    {}],
                                        8: [GaussianProcessClassifier,
                                                    {}],
                                        9: [GaussianNB,
                                                    {}],
                                        10: [BernoulliNB,
                                                    {}],
                                        11: [DecisionTreeClassifier,
                                                    {}],
                                        12: [ExtraTreesClassifier,
                                                    {}],
                                        13: [AdaBoostClassifier,
                                                    {}]
                                        }




df_prediction = pd.DataFrame({})
for keys, tuple_class_models in dict_list_class_ml_classifier_models.items():
    class_model = tuple_class_models[0]
    dict_list_hyperparameters = tuple_class_models[1]
    
    print(class_model.__name__)
    
    df_temp = func_run_model(x_train = x_train,
                                    x_test = x_test,
                                    y_train = y_train,
                                    y_test = y_test,
                                    bool_include_x_train_independent_variables = False,
                                    class_machine_learning_model = class_model,
                                    **dict_list_hyperparameters)

    df_prediction = df_prediction.append(df_temp,
                                         ignore_index=True)
    
df_prediction['RunDate'] = datetime.today().strftime('%Y-%m-%d')
df_prediction['RunTime'] = datetime.today().strftime('%H:%M:%S')
df_prediction.to_csv('Output.csv')




# %%

