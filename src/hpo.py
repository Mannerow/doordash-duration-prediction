"""
hpo.py

This script performs hyperparameter optimization (HPO) for selected models using Hyperopt.
It uses MLflow to log hyperparameters and performance metrics for each trial.
Models included are:
- RandomForestRegressor
- XGBRegressor

Usage:
    python src/hpo.py --data_path ./output --num_trials 15
"""

RANDOM_STATE = 42

import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("model-hyperopt")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="../data/processed_data",
    help="Location where the processed door dash data is saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="Number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
        model_type = params['type']
        del params['type']

        with mlflow.start_run():
            if model_type == 'RandomForestRegressor':
                mlflow.set_tag("model", "RandomForestRegressor")
                model = RandomForestRegressor(**params)
            elif model_type == 'XGBRegressor':
                mlflow.set_tag("model", "XGBRegressor")
                model = XGBRegressor(**params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            mlflow.log_params(params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = hp.choice('model_type', [
        {
            'type': 'RandomForestRegressor',
            'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
            'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
            'random_state': RANDOM_STATE
        },
        {
            'type': 'XGBRegressor',
            'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'subsample': hp.uniform('subsample', 0.7, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
            'random_state': RANDOM_STATE
        }
    ])

    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )

if __name__ == '__main__':
    run_optimization()
