#%% Import module

import unittest
import machine_learning_models
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#%%

class test_machine_learning_models(unittest.TestCase):
    
    def __init__(self):
        
        x, y = make_classification(n_samples=100, random_state=1)
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        self.x_train, self.x_test, self.y_train,self. y_test = train_test_split(x,y, 
                                                                                train_size = 0.50,
                                                                                shuffle= False)
        return None

    def test_model_01_classifier_random_forest_classifier(self):
        
        dict_model_results = machine_learning_models.model_01_classifier_random_forest_classifier.func_run_model(x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test,
                                                    int_random_forest_number_of_trees_estimator = 1000,
                                                    int_random_forest_max_dept = 5)
        
        self.assertEqual(type(dict_model_results,'dict'))
    
    
    

if __name__ == '__main__':
    unittest.main()
    
# %%
