import datetime
import logging 
import pandas as pd
import sys
import click
import boto3
import joblib
import psycopg
from scipy.sparse import csr_matrix

from botocore.exceptions import ClientError
from mlflow.tracking import MlflowClient
import mlflow.pyfunc

# Monitoring Imports
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, RegressionQualityMetric

# Load environment variables
from dotenv import load_dotenv
import os
import random
import uuid

import utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float,
    rmse float
)
"""

target = 'delivery_duration'

def init_mlflow():
    logging.info("Initializing MLflow...")
    load_dotenv()  # This will load all the env variables from the .env file
    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    logging.info("MLflow initialized.")

def load_best_model(best_model_bucket, best_model_name, experiment_name):
    logging.info("Loading best model from MLflow...")
    client = MlflowClient()
    # Get experiment ID from the experiment name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No experiment found with name '{experiment_name}'")
    
    experiment_id = experiment.experiment_id
    # Get all runs for the experiment
    runs = client.search_runs(experiment_ids=[experiment_id], order_by=["metrics.test_rmse ASC"], max_results=1)
    if not runs:
        raise ValueError("No runs found for the given experiment ID.")
    # Get the most recent run
    most_recent_run = runs[0]
    run_id = most_recent_run.info.run_id
    # Construct the S3 path using the experiment ID and run ID
    logged_model = f"s3://{best_model_bucket}/{experiment_id}/{run_id}/artifacts/model"
    logging.info(f"Logged Model = {logged_model}")
    model = mlflow.pyfunc.load_model(logged_model)
    logging.info("Best model loaded.")
    return model, run_id

def setup_monitoring(test_data_path: str, best_model_bucket: str, best_model_name: str, experiment_name: str):
    logging.info("Setting up monitoring...")
    init_mlflow()
    # Reference Data
    X_val, y_val = utils.load_pickle("../data/processed_data/val.pkl")
    logging.info(f'Reading the prepared data from {os.path.join(test_data_path, "test.pkl")}...')
    X_test, y_test = utils.load_pickle(os.path.join(test_data_path, "test.pkl"))
    logging.info(f'Loading the model from bucket={best_model_bucket}...')
    model, run_id = load_best_model(best_model_bucket, best_model_name, experiment_name)
    logging.info('Applying the model...')
    # X_test = X_test.fillna(0)

    y_pred = model.predict(X_test)
    y_val_pred = model.predict(X_val)

    logging.info(f"y_pred shape = {y_pred.shape}")
    dv = utils.load_pickle("../data/processed_data/dv.pkl")

    logging.info("Decoding Dataframes...")
    current_data = utils.decode_dataframe(dv, X_test)
    reference_data = utils.decode_dataframe(dv, X_val)

    # Get num and cat features
    num_features = current_data.select_dtypes(include=['number']).columns.tolist()
    cat_features = current_data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Set up target column
    current_data[target] = y_test
    reference_data[target] = y_val

    # Set up prediction column
    current_data['prediction'] = y_pred
    reference_data['prediction'] = y_val_pred

    



    print(f"Current Shape = {current_data.shape}")
    print(f"Ref Shape = {reference_data.shape}")

    # Column mapping
    column_mapping = ColumnMapping(
        target=target,
        prediction='prediction',
        numerical_features=num_features,
        categorical_features=cat_features
    )
    # Define the Evidently report
    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        RegressionQualityMetric()
    ])
    logging.info("Monitoring setup completed.")

    logging.info(f"Current Data = {current_data.head()}")
    logging.info(f"Reference Data = {reference_data.head()}")
    return current_data, reference_data, report, column_mapping

def prep_db():
    logging.info("Preparing database...")
    try:
        # Connect to PostgreSQL and create the database and table if they do not exist
        with psycopg.connect("host=db port=5432 user=postgres password=example", autocommit=True) as conn:
            res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
            if len(res.fetchall()) == 0:
                conn.execute("create database test;")
            with psycopg.connect("host=db port=5432 dbname=test user=postgres password=example") as conn:
                conn.execute(create_table_statement)
        logging.info("Database preparation completed.")
    except Exception as e:
        logging.error(f"Failed to prepare database: {e}")
        raise

def calculate_metrics_postgresql(current_data, reference_data, report, column_mapping):
    logging.info("Calculating metrics and storing in PostgreSQL...")
    # Run the Evidently report
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    # Extract metrics from the report
    result = report.as_dict()
    # logging.info(f"Report results: {result}")

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
    rmse = result['metrics'][3]['result']['current']['rmse']
    # Store metrics in PostgreSQL
    timestamp = datetime.datetime.now()
    with psycopg.connect("host=db port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
        with conn.cursor() as curr:
            curr.execute(
                "insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values, rmse) values (%s, %s, %s, %s, %s)",
                (timestamp, prediction_drift, num_drifted_columns, share_missing_values, rmse)
            )
    logging.info("Metrics successfully inserted into PostgreSQL.")

@click.command()
@click.option(
    "--test_data_path",
    default="../data/processed_data",  # path of test data
    help="Location where the raw DoorDash data is saved"
)
@click.option(
    "--best_model_bucket",
    default="mlflow-models-mannerow", 
    help="Bucket where the model is stored"
)
@click.option(
    "--best_model_name",
    default="doordash_best_model", 
    help="Name of the model to load"
)
@click.option(
    "--experiment_name",
    default="best-models", 
    help="Name of the experiment to load the model from"
)
def run(test_data_path: str, best_model_bucket: str, best_model_name: str, experiment_name: str):
    try:
        logging.info("Starting the run process...")
        prep_db()
        current_data, reference_data, report, column_mapping = setup_monitoring(test_data_path, best_model_bucket, best_model_name, experiment_name)  
        calculate_metrics_postgresql(current_data, reference_data, report, column_mapping)
        logging.info("Run process completed successfully.")
    except Exception as e:
        logging.error(f"Run process failed: {e}")
        raise

if __name__ == '__main__':
    run()
