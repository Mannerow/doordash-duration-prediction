import pickle
import mlflow
import pandas as pd
from scipy.sparse import csr_matrix
import sys
import click
import uuid

from mlflow.tracking import MlflowClient
import mlflow.pyfunc

# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()  # This will load all the env variables from the .env file

# Fetch the tracking URI from environment variables
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')

# Set the MLflow tracking URI
mlflow.set_tracking_uri(mlflow_tracking_uri)

def csr_to_dataframe(csr, feature_names):
    """Converts a CSR (Compressed Sparse Row) matrix to a Pandas DataFrame."""
    return pd.DataFrame.sparse.from_spmatrix(csr, columns=feature_names)

def generate_uuids(n):
    return [str(uuid.uuid4()) for _ in range(n)]

def load_pickle(filename: str, chunk_size=10000):
    """Yield chunks of data."""
    with open(filename, "rb") as f_in:
        while True:
            try:
                yield pickle.load(f_in)
            except EOFError:
                break

def load_most_recent_model(model_bucket, model_name):
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    most_recent_version = max(model_versions, key=lambda version: version.creation_timestamp)
    logged_model = f"s3://{model_bucket}/{most_recent_version.version}/{most_recent_version.run_id}/artifacts/model"
    print(f"Logged Model = {logged_model}")
    model = mlflow.pyfunc.load_model(logged_model)
    return model, most_recent_version.run_id

def save_results(df, y_pred, y_test, run_id, output_file):
    trip_ids = generate_uuids(len(y_test))  # Generating IDs based on the length of y_test
    results_df = pd.DataFrame({
        'trip_id': trip_ids,
        'actual_duration': y_test,
        'predicted_duration': y_pred,
        'diff': y_test - y_pred,
        'model_version': [run_id] * len(y_test)
    })
    
    feature_names = [
        'market_id',
        'store_id',
        'store_primary_category',
        'order_protocol',
        'total_items',
        'subtotal',
        'num_distinct_items',
        'min_item_price',
        'max_item_price',
        'total_onshift_dashers',
        'total_busy_dashers',
        'total_outstanding_orders',
        'estimated_order_place_duration',
        'estimated_store_to_consumer_driving_duration'
    ]

    # Check and convert csr_matrix to DataFrame
    if isinstance(df, csr_matrix):
        df = csr_to_dataframe(df, feature_names)
    
    final_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
    final_df.to_parquet(output_file, index=False)

def apply_model(test_data_path: str, model_bucket: str, model_name: str, dest_bucket: str):
    model, run_id = load_most_recent_model(model_bucket, model_name)
    for X_test_chunk, y_test_chunk in load_pickle(os.path.join(test_data_path, "test.pkl")):
        y_pred_chunk = model.predict(X_test_chunk)
        output_file = f's3://{dest_bucket}/{run_id}_chunk_{uuid.uuid4()}.parquet'
        print(f'Saving the result to {output_file}...')
        save_results(X_test_chunk, y_pred_chunk, y_test_chunk, run_id, output_file)

@click.command()
@click.option("--test_data_path", default="../data/processed_data", help="Location where the raw DoorDash data is saved")
@click.option("--model_bucket", default="mlflow-models-mannerow", help="S3 bucket where the models are stored")
@click.option("--model_name", default="doordash_best_model", help="Name of the best model in MLflow")
@click.option("--dest_bucket", default="doordash-duration-prediction-mannerow", help="S3 bucket where results will be saved")
def run(test_data_path: str, model_bucket: str, model_name: str, dest_bucket: str):
    apply_model(test_data_path, model_bucket, model_name, dest_bucket)

if __name__ == '__main__':
    run()
