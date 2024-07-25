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

data_preprocess.py
This script handles the preprocessing of raw data sourced from Kaggle. Key tasks include data cleaning, feature engineering, and splitting the dataset into training and testing subsets.

train.py
Responsible for training the machine learning model, this script encompasses model selection, training, and evaluation to ensure robust performance on unseen data.

hpo.py
This script performs hyperparameter optimization, fine-tuning the model’s performance by employing techniques like grid search or random search to determine the optimal set of hyperparameters.

register_model.py
Post-training, this script registers the final model with the MLflow tracking server, enabling versioning and easy retrieval for future predictions.

run_flow.py
This script initiates and executes the Prefect flow, orchestrating the entire machine learning pipeline from data preprocessing to model registration, ensuring each step is executed sequentially and correctly.

monitor_metrics.py
Designed for performance monitoring, this script tracks various metrics over time, helping identify any drifts or degradations in the model's performance.

score_batch.py
Used for batch scoring, this script allows the model to make predictions on new data batches. It can be scheduled to run at regular intervals, ensuring predictions remain current.

utils.py
This script contains utility functions that are utilized across different project scripts, promoting modularity and reusability within the codebase.