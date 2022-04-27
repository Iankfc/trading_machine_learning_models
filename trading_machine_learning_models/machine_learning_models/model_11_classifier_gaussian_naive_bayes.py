#%%

from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import make_classification
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
try:
    from .module_dict_model_result import func_dict_model_results
except ImportError:
    from module_dict_model_result import func_dict_model_results
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#%%

def func_run_model(x_train = None,
                    x_test = None,
                    y_train = None,
                    y_test = None,
                    **kwargs):
    
    """
    https://scikit-learn.org/stable/modules/naive_bayes.html
    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB

    """
    model = GaussianNB()

    model.fit(X = x_train, y = y_train)
    
    
    #df_feature_importance = pd.DataFrame({'Columns':x_train.columns,
    #                                        'Importance':model.feature_importances_}).sort_values('Importance', ascending= False)
    
    prediction = model.predict(X = x_test)
    #prediction_probability = pd.DataFrame(model.predict_proba(X = x_test))



    #%%
    df_prediction = pd.DataFrame(y_test.copy())

    df_prediction['Prediction'] = prediction

    #df_prediction['Probability of being 0'] = prediction_probability[0].to_list()
    #df_prediction['Probability of being 1'] = prediction_probability[1].to_list()



    #%%
    df_prediction = pd.merge(df_prediction, x_test, how = 'left', left_index= True, right_index= True)



    #%%
    print('Gausian Naive Bayes')
    print(confusion_matrix(y_true = y_test, y_pred = prediction))

    float_model_classification_accuracy = accuracy_score(y_true= y_test, y_pred=prediction)
    print(float_model_classification_accuracy)

    #plt.barh(df_feature_importance.Columns, df_feature_importance.Importance)
        
    
    dict_model_results =  func_dict_model_results(float_accuracy_score = accuracy_score,
                                            df_prediction = df_prediction,
                                            #df_feature_importance = df_feature_importance
                                            )

    return dict_model_results

#%%

if __name__ == '__main__':
    x, y = make_classification(n_samples=100, random_state=1)
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                                    train_size = 0.50,
                                                    shuffle= False)
    
    dict_model_results = func_run_model(x_train = x_train,
                                        x_test = x_test,
                                        y_train = y_train,
                                        y_test = y_test)

    df_prediction = dict_model_results['df_prediction']
    df_feature_importance =  dict_model_results['df_feature_importance']
    accuracy_score =  dict_model_results['float_accuracy_score']
