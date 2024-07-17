import os
import pickle
import click
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

from sklearn.feature_extraction import DictVectorizer


# def dump_pickle(obj, filename: str):
#     with open(filename, "wb") as f_out:
#         return pickle.dump(obj, f_out)


# def read_dataframe(filename: str):
#     df = pd.read_parquet(filename)

#     df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
#     df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
#     df = df[(df.duration >= 1) & (df.duration <= 60)]

#     categorical = ['PULocationID', 'DOLocationID']
#     df[categorical] = df[categorical].astype(str)

#     return df


# def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
#     df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
#     categorical = ['PU_DO']
#     numerical = ['trip_distance']
#     dicts = df[categorical + numerical].to_dict(orient='records')
#     if fit_dv:
#         X = dv.fit_transform(dicts)
#     else:
#         X = dv.transform(dicts)
#     return X, dv


def download_data(raw_data_path):
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Create the 'raw_data_path' folder if it doesn't exist
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)

    # Download the dataset to the 'raw_data_path' folder
    api.dataset_download_files('dharun4772/doordash-eta-prediction', path=raw_data_path, unzip=True)

    click.echo(f"Dataset downloaded to: {raw_data_path}")

@click.command()
@click.option(
    "--raw_data_path",
    default="../data",  # Default path for raw data
    help="Location where the raw DoorDash data is saved"
)
@click.option(
    "--dest_path",
    default="./data/processed_data",  # Default path for the resulting files
    help="Location where the resulting files will be saved"
)
def run_data_prep(raw_data_path: str, dest_path: str):

    download_data(raw_data_path=raw_data_path)


    # Load parquet files
    # df_train = read_dataframe(
    #     os.path.join(raw_data_path, f"{dataset}_tripdata_2023-01.parquet")
    # )
    # df_val = read_dataframe(
    #     os.path.join(raw_data_path, f"{dataset}_tripdata_2023-02.parquet")
    # )
    # df_test = read_dataframe(
    #     os.path.join(raw_data_path, f"{dataset}_tripdata_2023-03.parquet")
    # )

    # # Extract the target
    # target = 'duration'
    # y_train = df_train[target].values
    # y_val = df_val[target].values
    # y_test = df_test[target].values

    # # Fit the DictVectorizer and preprocess data
    # dv = DictVectorizer()
    # X_train, dv = preprocess(df_train, dv, fit_dv=True)
    # X_val, _ = preprocess(df_val, dv, fit_dv=False)
    # X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # # Create dest_path folder unless it already exists
    # os.makedirs(dest_path, exist_ok=True)

    # # Save DictVectorizer and datasets
    # dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    # dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    # dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    # dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == '__main__':
    run_data_prep()