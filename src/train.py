"""
train.py

This script trains and evaluates baseline models without hyperparameter tuning.
It uses MLflow to log model performance and metrics. Models included are:
- LinearRegression
- RandomForestRegressor
- XGBRegressor

Usage:
    python src/train.py --data_path ./output
"""

RANDOM_STATE = 42

import os
import pickle
import click
import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="../data/processed_data",
    help="Location where the processed door dash data is saved"
)
def run_train(data_path: str):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(max_depth=10, random_state=RANDOM_STATE),
        "XGBRegressor": XGBRegressor(max_depth=10, random_state=RANDOM_STATE)
    }

    for model_name, model in models.items():
        if model_name == "XGBRegressor":
            mlflow.xgboost.autolog()
        else:
            mlflow.sklearn.autolog()

        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("Developer", "Michael Mannerow")
            mlflow.set_tag("model", model_name)
            
            X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
            X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

            print(f"{model_name} RMSE: {rmse}")

if __name__ == '__main__':
    run_train()
