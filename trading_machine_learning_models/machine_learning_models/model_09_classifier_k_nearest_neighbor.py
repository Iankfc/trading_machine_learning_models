#%%

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
from .module_dict_model_result import func_dict_model_results



#%%

def func_run_model(x_train = None,
                    x_test = None,
                    y_train = None,
                    y_test = None):
    
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

    """
    model = KNeighborsClassifier()

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
    print('K Nearest Neighbor')
    print(confusion_matrix(y_true = y_test, y_pred = prediction))

    float_model_classification_accuracy = accuracy_score(y_true= y_test, y_pred=prediction)
    print(float_model_classification_accuracy)

    #plt.barh(df_feature_importance.Columns, df_feature_importance.Importance)
        
    
    dict_model_results =  func_dict_model_results(float_accuracy_score = accuracy_score,
                                            df_prediction = df_prediction,
                                            #df_feature_importance = df_feature_importance
                                            )

    return dict_model_results