import pickle
import mlflow
import pandas as pd
from scipy.sparse import csr_matrix
import sys
import click
import uuid
import boto3
from botocore.exceptions import ClientError
import utils

from mlflow.tracking import MlflowClient
import mlflow.pyfunc

# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()  # This will load all the env variables from the .env file

region = os.getenv('AWS_DEFAULT_REGION')

# Set the MLflow tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

def load_most_recent_model(model_bucket, model_name, experiment_name):
    client = MlflowClient()

    # Get experiment ID from the experiment name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No experiment found with name '{experiment_name}'")
    
    experiment_id = experiment.experiment_id

    # Get all runs for the experiment
    runs = client.search_runs(experiment_ids=[experiment_id], order_by=["metric.test_rmse ASC"], max_results=1)

    if not runs:
        raise ValueError("No runs found for the given experiment ID.")

    # Get the most recent run
    most_recent_run = runs[0]
    run_id = most_recent_run.info.run_id

    # Construct the S3 path using the experiment ID and run ID
    logged_model = f"s3://{model_bucket}/{experiment_id}/{run_id}/artifacts/model"
    print(f"Logged Model = {logged_model}")
    model = mlflow.pyfunc.load_model(logged_model)
    return model, run_id


def apply_model(test_data_path:str, model_bucket:str, model_name:str, dest_bucket:str):
    print(f'Reading the prepared data from {os.path.join(test_data_path, "test.pkl")}...')
    X_test, y_test = utils.load_pickle(os.path.join(test_data_path, "test.pkl"))

    print(f'Loading the model from bucket={model_bucket}...')
    model, run_id = load_most_recent_model(model_bucket, model_name, 'best-models')

    print('Applying the model...')
    y_pred = model.predict(X_test)
    print(f"y pred shape = {y_pred.shape}")

@click.command()
@click.option(
    "--test_data_path",
    default="../data/processed_data",  #path of test data
    help="Location where the raw DoorDash data is saved"
)
@click.option(
    "--model_bucket",
    default="mlflow-models-mannerow", 
    help="Location where the raw DoorDash data is saved"
)
@click.option(
    "--model_name",
    default="doordash_best_model", 
    help="Location where the raw DoorDash data is saved"
)
def run(test_data_path:str, model_bucket:str, model_name:str, dest_bucket:str):
    apply_model(test_data_path, model_bucket, model_name, dest_bucket)    

if __name__ == '__main__':
    run()
