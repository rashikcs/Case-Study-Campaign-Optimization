import pandas as pd
import numpy as np
import os
import pickle
from scripts.utils import create_directory
from scripts.utils import read_list_from_text

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier

def get_model(**kwargs)->object:
    """
    Loads and returns the desired model to train further with tuned parameters 
    found through HPO. All values passed as keyword arguments. 
    Must contain model name in the kawargs. 
    """
    if kwargs['name'].lower()=='xgboost':
        return XGBClassifier(colsample_bylevel= kwargs['colsample_bylevel'],
                            colsample_bytree= kwargs['colsample_bytree'], 
                            gamma= kwargs['gamma'],
                            learning_rate= kwargs['learning_rate'],
                            max_delta_step= kwargs['max_delta_step'], 
                            max_depth= kwargs['max_depth'],
                            min_child_weight= kwargs['min_child_weight'],
                            n_estimators= kwargs['n_estimators'],
                            reg_alpha= kwargs['reg_alpha'], 
                            reg_lambda= kwargs['reg_lambda'],
                            scale_pos_weight= kwargs['scale_pos_weight'],
                            subsample= kwargs['subsample'],
                            use_label_encoder=False,
                            random_state=kwargs['random_state'])
    
    elif kwargs['name'].lower()=='lightgbm':
        return LGBMClassifier(colsample_bytree = kwargs['colsample_bytree'], 
                                  learning_rate = kwargs['learning_rate'], 
                                  max_depth = kwargs['max_depth'], 
                                  n_estimators = kwargs['n_estimators'], 
                                  reg_alpha = kwargs['reg_alpha'], 
                                  reg_lambda = kwargs['reg_lambda'],
                                  scale_pos_weight  = kwargs['scale_pos_weight'], 
                                  boosting_type = kwargs['boosting_type'], 
                                  subsample = kwargs['subsample'], 
                                  num_leaves = kwargs['num_leaves'],
                                  random_state=kwargs['random_state'])

    elif kwargs['name'].lower()=='randomforest':
            return RandomForestClassifier(n_estimators = kwargs['n_estimators'], 
                                          max_depth = kwargs['max_depth'],
                                          max_features = kwargs['max_features'],
                                          min_samples_split=kwargs['min_samples_split'], 
                                          class_weight = kwargs['class_weight'],
                                          criterion= kwargs['criterion'],
                                          random_state=kwargs['random_state'])

    elif kwargs['name'].lower()=='ensemble':
            return VotingClassifier(estimators = kwargs['estimators'], 
                                    voting = kwargs['voting'],
                                    weights = kwargs['weights'])

def get_prediction(model:object, data:pd.core.frame.DataFrame)->list:
    """
    Returns prediction as list given a model object.
    """
    predict_proba = model.predict_proba(data)[:, 1]
    return predict_proba

def save_model(model:object, path:str, model_name:str, extension:str='pickle')->None:
    """
    saves the trained model as pickle file in the passed path.
    """
    try:
        if extension=='pickle':
            create_directory(path)
            model_name += ".pickle.dat"
            directory = os.path.join(path, model_name)

            pickle.dump(model, open(directory, "wb"))
            print('{} saved in : {}'.format(model_name, path))
        else:
            raise NotImplementedError
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))
        
def load_saved_model(path:str, model_name:str, extension:str='pickle')->object:
    """
    Loads the trained model saved as a pickle file.
    Args:
        path:str              -> directory to search
        model_name:str        -> model name to look for
        extension:str         -> saved extension

    """
    try: 
        if extension=='pickle':
            directory = os.path.join(path, model_name)
            print('{} found at {}.'.format(model_name, directory))
            return pickle.load(open(directory+".pickle.dat", "rb"))
        else:
            NotImplementedError
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def get_feature_list_from_text_excluding_target(feature_list_path:str,
                                                target_column:str)->list:
    """
    Reads list fro the given text file. Removes target column from it and
    returns the list

    """  

    try:
        feature_list = read_list_from_text(feature_list_path)
        feature_list.sort(reverse=True)
        
        if target_column in feature_list:
            feature_list.remove(target_column)
            
        return feature_list
    
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))
