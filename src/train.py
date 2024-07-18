"""
train.py

This script trains and evaluates baseline models without hyperparameter tuning.
It uses MLflow to log model performance and metrics. Models included are:
- LinearRegression
- XGBRegressor
- Ridge

Usage:
    python src/train.py --data_path ./output
"""

RANDOM_STATE = 42

import os
import pickle
import click
import mlflow

from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("model-train")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="../data/processed_data",
    help="Location where the processed DoorDash data is saved"
)
def run_train(data_path: str):
    models = {
        "LinearRegression": LinearRegression(),
        "XGBRegressor": XGBRegressor(max_depth=5, n_estimators=50, random_state=RANDOM_STATE),
        "Ridge": Ridge(alpha=0.5, random_state=RANDOM_STATE)  # Adjusted alpha for less complexity
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("Developer", "Michael Mannerow")
            mlflow.set_tag("model", model_name)
            
            X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
            X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

            # Log model parameters
            if model_name == "XGBRegressor":
                mlflow.log_params({"max_depth": 5, "n_estimators": 50, "random_state": RANDOM_STATE})
            elif model_name == "Ridge":
                mlflow.log_params({"alpha": 0.5, "random_state": RANDOM_STATE})
            # LinearRegression does not have hyperparameters in this example

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

            print(f"{model_name} RMSE: {rmse}")

if __name__ == '__main__':
    run_train()
99