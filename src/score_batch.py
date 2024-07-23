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

def generate_uuids(n):
    trip_ids = []
    for i in range(n):
        trip_ids.append(str(uuid.uuid4()))
    return trip_ids

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

def save_results(df, y_pred, y_test, run_id, output_file):
    trip_ids = generate_uuids(df.shape[0])
    results_df = pd.DataFrame({
        'trip_id': trip_ids,
        'actual_duration': y_test,
        'predicted_duration': y_pred,
        'diff': y_test - y_pred,
        'model_version': run_id
    })
    
    dv = utils.load_pickle("../data/processed_data/dv.pkl")

    # Ensure df is a DataFrame, not a csr_matrix
    if isinstance(df, csr_matrix):
        # Convert sparse matrix to dense matrix
        X_dense = df.toarray()
        
        # Retrieve feature names from DictVectorizer
        feature_names = dv.get_feature_names_out()
        
        # Create DataFrame from dense matrix
        df_dense = pd.DataFrame(X_dense, columns=feature_names)
    else:
        df_dense = df
    
    final_df = pd.concat([df_dense.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

    # Print the first few rows of the final DataFrame for debugging
    print("First few rows of the final DataFrame for debugging:")
    print(final_df.head())

    final_df.to_parquet(output_file, index=False)

def create_s3_bucket(bucket_name, region=None):
    # Initialize a session using Amazon S3
    s3_client = boto3.client('s3', region_name=region)
    
    # Check if the bucket exists
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except ClientError as e:
        # If the bucket does not exist, create it
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            try:
                if region is None or region == 'us-east-1':
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                print(f"Bucket '{bucket_name}' created successfully.")
            except ClientError as e:
                print(f"Error occurred while creating bucket: {e}")
        else:
            print(f"Error occurred: {e}")

def apply_model(test_data_path:str, model_bucket:str, model_name:str, dest_bucket:str):
    print(f'Reading the prepared data from {os.path.join(test_data_path, "test.pkl")}...')
    X_test, y_test = utils.load_pickle(os.path.join(test_data_path, "test.pkl"))

    print(f'Loading the model from bucket={model_bucket}...')
    model, run_id = load_most_recent_model(model_bucket, model_name, 'best-models')

    print('Applying the model...')
    y_pred = model.predict(X_test)
    print(f"y pred shape = {y_pred.shape}")

    print(f"Region = {region}")
    create_s3_bucket(dest_bucket, region)

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
