import os
import json

def save_tuned_result(model_name:str, info:dict, path:str)->None:

    try:
        directory = os.path.join(path, model_name)
        create_directory(directory)
        file = open(f"{directory}{os.sep}best_params.json", "w")
        json.dump(info, file)
        file.close()
        print('Successfully saved results in', directory)
    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))
        
def get_tuned_result(model_name:str, path:str)->dict:

    try:
        directory = os.path.join(path, model_name)
        file = open(f"{directory}{os.sep}best_params.json", "r") 

        return json.loads(file.read())

    except Exception as error:
        raise Exception('Caught this error: ' + repr(error))