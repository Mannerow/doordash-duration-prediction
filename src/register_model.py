"""Analyzes models based on performance metrics, then registers best model to MLFlow"""

import os

import click
import mlflow
from dotenv import load_dotenv
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import utils

load_dotenv()  # This method reads the .env file and loads it into the environment

HPO_EXPERIMENT_NAME = "model-hyperopt"
TRAIN_EXPERIMENT_NAME = "model-train"
BEST_MODELS_EXPERIMENT_NAME = "best-models"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(BEST_MODELS_EXPERIMENT_NAME)


def train_and_log_model(data_path, params):
    """Trains models and logs to MLFlow"""
    X_train, y_train = utils.load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = utils.load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = utils.load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        model_type = params.pop("type")
        model = None
        # Convert parameters to the appropriate type
        if model_type == "XGBRegressor":
            params["max_depth"] = int(params["max_depth"])
            params["n_estimators"] = int(params["n_estimators"])
            params["random_state"] = int(params["random_state"])
            model = XGBRegressor(**params)
        elif model_type == "Ridge":
            params["alpha"] = float(params["alpha"])
            params["random_state"] = int(params["random_state"])
            model = Ridge(**params)
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, model.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        # rint(f"Logged val_rmse: {val_rmse}")
        test_rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)

        # Log the model
        mlflow.sklearn.log_model(model, artifact_path="model")


@click.command()
@click.option(
    "--data_path",
    default="../data/processed_data",
    help="Location where the processed DoorDash data is saved",
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote",
)
def run_register_model(data_path: str, top_n: int):
    """Loads best model and registers it."""
    client = MlflowClient()

    # Retrieve the top_n model runs and log the models/
    # Find best runs on validation data from HPO experiment
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    hpo_runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.val_rmse ASC"],
    )

    for run in hpo_runs:
        # print(f"Processing HPO run ID: {run.info.run_id} with params: {run.data.params}")
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Now, Select the model with the lowest val RMSE from train
    experiment = client.get_experiment_by_name(TRAIN_EXPERIMENT_NAME)
    train_runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.val_rmse ASC"],
    )

    # Train and log models from train runs to give them val and test rmse
    for run in train_runs:
        # print(f"Processing train run ID: {run.info.run_id} with params: {run.data.params}")
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Now get the best runs
    experiment = client.get_experiment_by_name(BEST_MODELS_EXPERIMENT_NAME)
    best_runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_rmse ASC"],
    )

    # Print metrics for debugging
    for run in best_runs:
        print(f"Run ID: {run.info.run_id}, Metrics: {run.data.metrics}")

    # # Find the best run based on test RMSE
    best_run = min(best_runs, key=lambda run: run.data.metrics["test_rmse"])

    # # Register the best
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="doordash_best_model")


if __name__ == "__main__":
    run_register_model()
