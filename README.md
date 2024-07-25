# doordash-duration-prediction

## 📄 Problem Description

The objective of this project is to predict the delivery duration for DoorDash orders using historical data. This involves developing a machine learning model that can accurately forecast the time it will take for a delivery to reach the customer from the moment an order is placed. The project will leverage MLOps principles to ensure robust experimentation, model tracking, monitoring, and automation throughout the development and deployment lifecycle.

## 📊 Dataset

The dataset used for this project is sourced from Kaggle and contains various features related to DoorDash orders, such as pickup and drop-off locations, timestamps, order size, and other relevant attributes. The dataset can be found at DoorDash ETA Prediction Dataset.

## 🔒 Authentication

This project requires Kaggle, AWS, and Prefect accounts. Kaggle is used to download the dataset, AWS is used to store the model and predictions, and Prefect is used for pipeline orchestration.

1. Place your kaggle.json in the root directory of this project.
2. Create a .env file which will contain the necessary environment variables for AWS and Prefect authentication and the MLFlow Endpoint.

```bash
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_DEFAULT_ARTIFACT_ROOT='PATH-GOES-HERE'
MLFLOW_BACKEND_STORE_URI='sqlite:///backend.db'
AWS_ACCESS_KEY_ID='YOUR-ACCESS-KEY'
AWS_SECRET_ACCESS_KEY='YOUR-SECRET-ACCESS-KEY'
AWS_DEFAULT_REGION=us-east-1
PREFECT_API_KEY='YOUR-PREFECT-API-KEY'
PREFECT_WORKSPACE='YOUR-PREFECT-WORKSPACE-NAME'
```

## ⚙️ Installation

To get started with the project, follow these steps:

**1. Clone the repository:**

```bash
git clone https://github.com/Mannerow/doordash-duration-prediction
cd doordash-duration-prediction
```

**2. Install the required dependencies:**

```bash
pip install pipenv
pipenv install
```

## 🚀 Running the Project

To run the project, simply build and run the docker image: 

```bash
docker compose up
```

## 🌐 Viewing Prefect Cloud

The app service provides a link to Prefect Cloud. Open this link to access the Prefect Cloud UI, which offers detailed insights into deployments and previous runs. While the deployment is scheduled to run hourly, you can manually initiate a run by clicking 'Run' for immediate execution.

## 🔍 How It Works

This section provides an overview of the primary scripts and their functionalities within the project, illustrating the workflow from data preprocessing to model deployment and monitoring:

**`experimentation.ipynb`**

This Jupyter notebook is used for exploratory data analysis (EDA) and initial experimentation with the DoorDash delivery duration prediction dataset. It imports necessary packages and downloads the dataset from Kaggle. The notebook includes steps for data exploration, such as loading the dataset into a Pandas DataFrame, checking for missing values, and creating new features like delivery duration. It also performs data visualization to identify outliers and understand data distributions. The notebook demonstrates the process of data cleaning by removing outliers and preparing features for modeling. It concludes with building a simple linear regression model using a pipeline, splitting the data into training and test sets, and evaluating model performance using metrics like RMSE and MAE. This notebook serves as an interactive environment to explore and experiment with different approaches before formalizing them into the main project pipeline.

**`run_flow.py`**

This script initiates and executes the Prefect flow, orchestrating the entire machine learning pipeline from data preprocessing to model registration. It ensures each step is executed sequentially and correctly, running the pipeline on an hourly schedule.

**`data_preprocess.py`**

This script prepares raw data for machine learning using `pandas`, `scikit-learn`, and the `Kaggle API`. It reads the raw data, creates a delivery duration feature, handles missing values, and removes outliers. The script then extracts relevant features, vectorizes them, and splits the data into training, validation, and test sets. This ensures the data is clean and well-structured for model training and evaluation.

**`train.py`**

This script trains and evaluates baseline models without hyperparameter tuning. It leverages `MLflow` to log model performance and metrics for models such as `LinearRegression`, `XGBRegressor`, and `Ridge`. The script reads processed data, trains each model, evaluates it using RMSE, and logs the results to `MLflow`. This ensures reproducibility and easy tracking of model performance.

**`hpo.py`**

This script performs hyperparameter optimization (HPO) for selected models using `Hyperopt` and logs the hyperparameters and performance metrics for each trial using `MLflow`. Hyperparameter tuning involves systematically searching for the best set of parameters that improves the model's performance. The script optimizes models such as `Ridge` and `XGBRegressor`, enhancing their performance by exploring various hyperparameter configurations. It reads processed data, runs the optimization, and records the results to ensure efficient model tuning.

**`register_model.py`**

This script identifies and registers the best-performing models using `MLflow`. It evaluates models from hyperparameter optimization and training experiments based on validation and test RMSE. The script reads processed data, trains and evaluates models, and registers the best model to ensure reproducibility and ease of deployment. The best model is stored in the default artifact location specified in the .env file.

**`score_batch.py`**

This script applies the best-performing machine learning model to the test dataset and saves the prediction results to an S3 bucket. It reads the processed test data, loads the best model from the specified S3 bucket using `MLflow`, and generates predictions. Hyperparameter tuning systematically searches for the best set of parameters to improve model performance. The script handles the creation of the destination S3 bucket if it does not already exist. The results, including predicted and actual durations along with the model version, are saved as a Parquet file in the specified S3 bucket. The environment variables, such as MLFLOW_TRACKING_URI and AWS_DEFAULT_REGION, are loaded from the .env file.

**`monitor_metrics.py`**

TODO

**`utils.py`**

This script contains utility functions that are utilized across different project scripts, promoting modularity and reusability within the codebase. Specifically, it includes helper functions for loading and dumping pickle files.