import pandas as pd
import os


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

