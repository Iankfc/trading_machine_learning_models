#%% Import module

import unittest

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from machine_learning_models import model_01_classifier_random_forest_classifier as model01
from machine_learning_models import model_02_classifier_neural_net as model02
from machine_learning_models import model_03_classifier_gradient_boosting_classifier as model03
from machine_learning_models import model_04_classifier_support_vector_machine as model04
from machine_learning_models import model_05_classifier_quadratic_discriminant as model05
from machine_learning_models import model_06_classifier_linear_discriminant as model06
from machine_learning_models import model_07_classifier_ridge as model07
from machine_learning_models import model_08_classifier_sgd as model08
from machine_learning_models import model_09_classifier_k_nearest_neighbor as model09
from machine_learning_models import model_10_classifier_gaussian as model10
from machine_learning_models import model_11_classifier_gaussian_naive_bayes as model11
from machine_learning_models import model_12_classifier_bernoulli_naive_bayes as model12
from machine_learning_models import model_13_classifier_decision_tree as model13
from machine_learning_models import model_14_classifier_extra_trees as model14
from machine_learning_models import model_15_classifier_adaboost as model15

#%%

class test_machine_learning_models(unittest.TestCase):
    
    def setUp(self):
        #Arrange
        x, y = make_classification(n_samples=100, random_state=1)
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y, 
                                                                                train_size = 0.50,
                                                                                shuffle= False)
        
    def test_model01(self):
        
        #Act
        dict_hyperparameters = {'n_estimators':100,
                            'max_depth':5}
        
        dict_model_results = model01.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test,
                                                     **dict_hyperparameters)
        
        
   
    
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
    def test_model02(self):
        
        #Act
        
        dict_hyperparameters = {'solver':'lbfgs', 
                        'alpha':1e-5,
                        'hidden_layer_sizes':(5,2), 
                        'random_state':1}
            
        dict_model_results = model02.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test,
                                                    **dict_hyperparameters)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
        
    def test_model03(self):
        
        #Act
        dict_model_results = model03.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
    def test_model04(self):
        
        #Act
        dict_model_results = model04.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
    def test_model05(self):
        
        #Act
        dict_model_results = model05.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
    def test_model06(self):
        
        #Act
        dict_model_results = model06.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
    def test_model07(self):
        
        #Act
        dict_model_results = model07.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
    def test_model08(self):
        
        #Act
        dict_model_results = model08.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
    def test_model09(self):
        
        #Act
        dict_model_results = model09.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
    def test_model10(self):
        
        #Act
        dict_model_results = model10.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
    def test_model11(self):
        
        #Act
        dict_model_results = model11.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
    def test_model12(self):
        
        #Act
        dict_model_results = model12.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
    def test_model13(self):
        
        #Act
        dict_model_results = model13.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
    def test_model14(self):
        
        #Act
        dict_model_results = model14.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
    def test_model15(self):
        
        #Act
        dict_model_results = model15.func_run_model( x_train = self.x_train,
                                                    x_test = self.x_test,
                                                    y_train = self.y_train,
                                                    y_test = self.y_test)
        str_result = len(dict_model_results)
        
        #Assert
        self.assertEqual(str_result,3)
        
        
    
    
    

if __name__ == '__main__':
    unittest.main()
    

            
# %%
