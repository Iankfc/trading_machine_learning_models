#%%
from sklearn.model_selection import GridSearchCV 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')
#%%

def func_dict_grid_search(x_train = None,
                        y_train = None,
                        class_machine_learning_model = None,
                        dict_nparray_parameter_grid = None):

    grid = GridSearchCV(estimator = class_machine_learning_model,
                        param_grid = dict_nparray_parameter_grid,
                        cv = 5)

    grid.fit(X = x_train, y = y_train)

    df_grid_result = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)

    dict_best_hyperparameters = df_grid_result.sort_values('Accuracy', ascending=True).tail(1).reset_index(drop = True).drop(labels = ['Accuracy'],axis=1).T.to_dict()[0]
    
    return dict_best_hyperparameters
  
  
#%%

if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier 
    x, y = make_classification(n_samples=100, random_state=1)
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                    train_size = 0.50,
                                                    shuffle= False)
    
    nparray_n_estimators = np.arange(1,3,1)
    nparray_max_depth = np.arange(1,3,1)
    
    dict_nparray_hyperparameters = {'n_estimators':nparray_n_estimators,
                                    'max_depth':nparray_max_depth}
    
    dict_best_hyperparameters = func_dict_grid_search(x_train = x_train,
                                                        y_train = y_train,
                                                        class_machine_learning_model = RandomForestClassifier(),
                                                        dict_nparray_parameter_grid = dict_nparray_hyperparameters)
    

#%%


# %%
