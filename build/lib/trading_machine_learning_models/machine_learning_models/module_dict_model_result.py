def func_dict_model_results(float_accuracy_score = None,
                            df_prediction = None,
                            df_feature_importance = None):
  
  dict_results = {'float_accuracy_score':float_accuracy_score,
                  'df_prediction': df_prediction,
                  'df_feature_importance':df_feature_importance}
  
  return dict_results