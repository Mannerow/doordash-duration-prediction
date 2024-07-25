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

1. **Clone the repository:**

```bash
git clone https://github.com/Mannerow/doordash-duration-prediction
cd doordash-duration-prediction
```

2. **Install the required dependencies:**

```bash
pip install pipenv
pipenv install
```

## 🚀 Running the Project

To run the project, simply build and run the docker image: 

```bash
docker compose up
```