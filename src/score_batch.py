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

def csr_to_dataframe(csr_matrix, feature_names):
    """Converts a CSR (Compressed Sparse Row) matrix to a Pandas DataFrame."""
    df = pd.DataFrame(csr_matrix.toarray(), columns=feature_names)
    return df

def generate_uuids(n):
    trip_ids = []
    for i in range(n):
        trip_ids.append(str(uuid.uuid4()))
    return trip_ids

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def load_most_recent_model(model_bucket, model_name):
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    
    if not model_versions:
        raise ValueError("No model versions found for given model name.")

    # Sort versions by creation time, assuming version information includes creation timestamp
    most_recent_version = max(model_versions, key=lambda version: version.creation_timestamp)
    
    # Construct the S3 path using the retrieved most recent version
    logged_model = f"s3://{model_bucket}/{most_recent_version.version}/{most_recent_version.run_id}/artifacts/model"
    print(f"Logged Model = {logged_model}")
    model = mlflow.pyfunc.load_model(logged_model)
    return model, most_recent_version.run_id

def save_results(df, y_pred, y_test, run_id, output_file):
    trip_ids = generate_uuids(df.shape[0])
    results_df = pd.DataFrame({
        'trip_id': trip_ids,
        'actual_duration': y_test,
        'predicted_duration': y_pred,
        'diff': y_test - y_pred,
        'model_version': run_id
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

    print("new new")

    # Ensure df is a DataFrame, not a csr_matrix
    if isinstance(df, csr_matrix):
        df = csr_to_dataframe(df, feature_names)  # Convert csr_matrix to DataFrame if necessary
    
    final_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

    # Print the first few rows of the final DataFrame for debugging
    print("First few rows of the final DataFrame for debugging:")
    print(final_df.head())

    final_df.to_parquet(output_file, index=False)





def apply_model(test_data_path:str, model_bucket:str, model_name:str, dest_bucket:str):
    print(f'Reading the prepared data from {os.path.join(test_data_path, "test.pkl")}...')
    X_test, y_test = load_pickle(os.path.join(test_data_path, "test.pkl"))

    print(f'Loading the model from bucket={model_bucket}...')
    model, run_id = load_most_recent_model(model_bucket, model_name)

    print('Applying the model...')
    y_pred = model.predict(X_test)
    print(f"y pred shape = {y_pred.shape}")

    output_file = f's3://{dest_bucket}/{run_id}.parquet'
    print(f'Saving the result to {output_file}...')

    save_results(X_test, y_pred, y_test, run_id, output_file)
    return output_file

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
@click.option(
    "--dest_bucket",
    default="doordash-duration-prediction-mannerow",  # Default path for the resulting files
    help="Location where the resulting files will be saved"
)
def run(test_data_path:str, model_bucket:str, model_name:str, dest_bucket:str):
    apply_model(test_data_path, model_bucket, model_name, dest_bucket)    

if __name__ == '__main__':
    run()
