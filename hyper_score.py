
import os
import numpy as np
import json
import joblib
from azureml.core.model import Model

def init():
    global model
    try:
        model_path = Model.get_model_path('hyperdrive_best_run.pkl')
        model = joblib.load(model_path)
    except Exception as err:
        print("init method error: "+str(err))

def run(data):
    try:
        #data = np.array(json.loads(data))
        data = json.loads(data)
        strn = "extracted json\n"
        data = np.array(data["data"])
        strn += "converted data to np array\n"
        result = model.predict(data)
        strn += "sent the data to the model for prediction\n"
        print(strn)
        return result.tolist()
    except Exception as err:
        return strn+"run method error: "+str(err)
