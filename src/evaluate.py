

import pandas as pd

import pickle
from sklearn.metrics import accuracy_score

import yaml
import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Zeineb-ty/class_pipeline"
os.environ["MLFLOW_TRACKING_USERNAME"] = 'Zeineb-ty'
os.environ["MLFLOW_TRACKING_PASSWORD"]= "fb241dcd22c80e18dacb7b096f1d064f10c83a52"

# load
train_params = yaml.safe_load(open("params.yaml"))['train']
dagshub_params = yaml.safe_load(open("params.yaml"))['dagshub']


def evaluate(data_path,model_path):
    data =pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(dagshub_params["MLFLOW_TRACKING_URI"])

    #? model = pickle.load(open(model_path,'rb'))
    import joblib

    model = joblib.load(model_path)

    predictions = model.predict(X)
    accuracy = accuracy_score(y,predictions)

    mlflow.log_metric("eval_accuracy",accuracy)

    print(f"Model acccuracy :{accuracy}")

if __name__=="__main__":
    evaluate(train_params["data"],train_params["model_path"])