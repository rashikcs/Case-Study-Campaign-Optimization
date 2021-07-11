import pandas as pd
import numpy as np
import os
import pickle

def create_directory(directory:str)->None:
    """
    Given a directory this function checks and 
    creates creates directory if doesn't exist.
    
    """
    try: 
        if not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def save_list_as_text_file(path:str, features:list, extension:str='pickle')->None:
    
    try:
        if extension=='pickle':
            with open(path, "wb") as opened_file:   #Pickling
                pickle.dump(features, opened_file)
            print('List saved in: ', path)
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))

def read_list_from_text(path:str, extension:str='pickle')->list:
    
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