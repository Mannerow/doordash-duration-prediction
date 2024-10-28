"""
hpo.py

This script performs hyperparameter optimization (HPO) for selected models using Hyperopt.
It uses MLflow to log hyperparameters and performance metrics for each trial.
Models included are:
- Ridge
- XGBRegressor

Usage:
    python src/hpo.py --data_path ./output --num_trials 15
"""

RANDOM_STATE = 42

import os

import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import utils

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("model-hyperopt")


@click.command()
@click.option(
    "--data_path",
    default="../data/processed_data",
    help="Location where the processed DoorDash data is saved",
)
@click.option(
    "--num_trials",
    default=5,
    help="Number of parameter evaluations for the optimizer to explore",
)
def run_optimization(data_path: str, num_trials: int):
    """runs the optimizer"""

    X_train, y_train = utils.load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = utils.load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
        model_type = params.pop("type")  # Remove 'type' from params

        with mlflow.start_run():
            mlflow.set_tag("model", model_type)
            if model_type == "Ridge":
                model = Ridge(**params)
            elif model_type == "XGBRegressor":
                model = XGBRegressor(**params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            params["type"] = (
                model_type  # Add type again, need it for the register model script
            )
            mlflow.log_params(params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("val_rmse", rmse)

        return {"loss": rmse, "status": STATUS_OK}

    search_space = hp.choice(
        "model_type",
        [
            {
                "type": "XGBRegressor",
                "max_depth": scope.int(
                    hp.quniform("max_depth", 3, 6, 1)
                ),  # Reduced depth
                "n_estimators": scope.int(
                    hp.quniform("n_estimators", 10, 50, 10)
                ),  # Fewer trees
                "learning_rate": hp.uniform(
                    "learning_rate", 0.1, 0.3
                ),  # Higher learning rate
                "subsample": hp.uniform("subsample", 0.8, 1.0),
                "colsample_bytree": hp.uniform("colsample_bytree", 0.8, 1.0),
                "random_state": RANDOM_STATE,
            },
            {
                "type": "Ridge",
                "alpha": hp.loguniform(
                    "alpha", np.log(0.001), np.log(1)
                ),  # Narrower range for alpha
                "random_state": RANDOM_STATE,
            },
        ],
    )

    rstate = np.random.default_rng(RANDOM_STATE)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate,
    )


if __name__ == "__main__":
    run_optimization()
