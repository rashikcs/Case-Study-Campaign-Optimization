import pandas as pd
import numpy as np
import os
import pickle
import random
import json

def set_seed(seed:int)->None:
    """
    sets the passed seed value in numpy, os and random libraries.
    
    """ 
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_dict_as_json(path:str, info:dict, file_name:str= 'best_params')->None:
    """
    Given a directory this function loads and
    returns the file as dict.
    
    """ 
    try:
        create_directory(path)
        file = open(f"{path}{os.sep}{file_name}.json", "w")
        json.dump(info, file)
        file.close()
        print('Successfully saved info in', path)
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def read_json_file(path:str)->dict:
    """
    Given a directory this function loads json and
    returns the file as dict.
    
    """    
    try:
        if os.path.exists(path):
            file = open(f"{path}", "r") 
            return json.loads(file.read())

    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def create_directory(directory:str)->None:
    """
    Given a directory this function checks and 
    creates creates directory if it doesn't exist.
    
    """
    try: 
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def save_list_as_text_file(path:str, features:list, extension:str='pickle')->None:
    """
    saves the passed list as text file in the passed path.
    Args:
        save_directory:str   -> directory to save output
        extension:str        ->  extension to use for serializing
    """
    try:
        if extension=='pickle':
            with open(path, "wb") as opened_file:   #Pickling
                pickle.dump(features, opened_file)
            print('List saved in: ', path)
        else:
            NotImplementedError
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def read_list_from_text(path:str, extension:str='pickle')->list:
    """
    loads list from a text file.
    """    
    try:
        if extension=='pickle':
            with open(path, "rb") as opened_file:
                print('Successfully returned list from: ', path)
                return pickle.load(opened_file)
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))


def extract_date_features(prepared_df:pd.core.frame.DataFrame,
                          date_column:str='datetime',
                          feature_name:str='hour')->list:
    """
    Extract date features(i.e. hour, day of week).
    """
    try:
        prepared_df[date_column] = pd.to_datetime(prepared_df[date_column])

        if feature_name=='hour':
            return prepared_df['datetime'].dt.hour
        elif feature_name=='dayofweek':
            return prepared_df['datetime'].dt.dayofweek
        else:
            raise NotImplementedError
            
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))