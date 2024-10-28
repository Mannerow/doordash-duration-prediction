"""Reads the data, performs feature engineering, and splits data into training/testing sets."""

import os

import click
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import utils


def read_dataframe(raw_data_path: str):
    """Reads dataframe. Just uses half the data"""
    csv_file_path = os.path.join(raw_data_path, "historical_data.csv")

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Return a random sample of half the dataset (fixes OOM errors)
    half_df = df.sample(
        frac=0.50, random_state=1
    )  # random_state ensures reproducibility
    return half_df


def create_delivery_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Create delivery duration feature in minutes"""
    # Convert 'created_at' and 'actual_delivery_time' to datetime
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"])

    # Calculate delivery_duration in minutes
    df["delivery_duration"] = (
        df["actual_delivery_time"] - df["created_at"]
    ).dt.total_seconds() / 60
    return df


def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove outliers from column"""
    # Calculate summary statistics
    summary_stats = df[column].describe()

    # Calculate the interquartile range (IQR)
    Q1 = summary_stats["25%"]
    Q3 = summary_stats["75%"]
    IQR = Q3 - Q1

    # Define outlier thresholds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    filtered_df = df[
        (df["delivery_duration"] >= lower_bound)
        & (df["delivery_duration"] <= upper_bound)
    ]

    return filtered_df


def extract_features_and_target(df: pd.DataFrame, target: str):
    """Extract features and target from DataFrame"""
    categorical_columns = [
        "created_at",
        "actual_delivery_time",
        "store_primary_category",
    ]
    numerical_columns = [
        col for col in df.columns if col not in categorical_columns and col != target
    ]

    # Drop the target column from the DataFrame
    df_features = df.drop(columns=[target])

    # Convert only categorical columns to string
    df_features[categorical_columns] = df_features[categorical_columns].astype(str)

    # Convert DataFrame to dictionary records
    train_dicts = df_features.to_dict(orient="records")

    return train_dicts, df[target].astype(float)


def vectorize_and_split(train_dicts, target):
    """Vectorize features and split data into train, validation, and test sets. train: 0.8, val: 0.1, teset: 0.1"""
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(train_dicts)

    # Split the data into train and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, target, test_size=0.2, random_state=42
    )

    # Further split the test set into val and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, dv


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handles categorical data and drops nulls"""
    # Define imputer for numerical columns
    imputer = SimpleImputer(strategy="mean")
    numerical_columns = df.select_dtypes(include=["number"]).columns

    # Apply imputer to numerical columns
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    # Drop rows with missing categorical values
    df = df.dropna()

    return df


def preprocess(df: pd.DataFrame, target: str = "delivery_duration"):
    """Full preprocessing pipeline"""
    df = create_delivery_duration(df)
    df = handle_missing_values(df)
    df = remove_outliers(df, target)
    train_dicts, target = extract_features_and_target(df, target)
    X_train, X_val, X_test, y_train, y_val, y_test, dv = vectorize_and_split(
        train_dicts, target
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, dv


def download_data(raw_data_path: str):
    """Downloads raw data from Kaggle"""
    # Initialize the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Create the 'raw_data_path' folder if it doesn't exist
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)

    # Download the dataset to the 'raw_data_path' folder
    api.dataset_download_files(
        "dharun4772/doordash-eta-prediction", path=raw_data_path, unzip=True
    )

    click.echo(f"Dataset downloaded to: {raw_data_path}")


@click.command()
@click.option(
    "--raw_data_path",
    default="../data",  # Default path for raw data
    help="Location where the raw DoorDash data is saved",
)
@click.option(
    "--dest_path",
    default="../data/processed_data",  # Default path for the resulting files
    help="Location where the resulting files will be saved",
)
def run_data_prep(raw_data_path: str, dest_path: str):
    """Runs data prep pipeline and dumps the pickles locally."""

    download_data(raw_data_path=raw_data_path)

    df = read_dataframe(raw_data_path)

    # Preprocess the data
    X_train, X_val, X_test, y_train, y_val, y_test, dv = preprocess(
        df, target="delivery_duration"
    )

    # # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    print(f"Dumping Pickles to {dest_path}")
    # # Save DictVectorizer and datasets
    utils.dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    utils.dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    utils.dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    utils.dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == "__main__":
    run_data_prep()
