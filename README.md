# doordash-duration-prediction

## Problem Description

The objective of this project is to predict the delivery duration for DoorDash orders using historical data. This involves developing a machine learning model that can accurately forecast the time it will take for a delivery to reach the customer from the moment an order is placed. The project will leverage MLOps principles to ensure robust experimentation, model tracking, monitoring, and automation throughout the development and deployment lifecycle.

## Dataset

The dataset used for this project is sourced from Kaggle and contains various features related to DoorDash orders, such as pickup and drop-off locations, timestamps, order size, and other relevant attributes. The dataset can be found at DoorDash ETA Prediction Dataset.

## Authentication

This project requires a Kaggle and AWS account. Kaggle is used to download the dataset, and AWS is used to store the model and predictions. 

1. Place your kaggle.json in .kaggle folder to faciliate downloading the dataset.
2. Create a '.env' file which will contain the necessary variables for AWS authentication and the MLFlow Endpoint. 

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
AWS_ACCESS_KEY_ID='YOUR-ACCESS-KEY'
AWS_SECRET_ACCESS_KEY='YOUR-SECRET-ACCESS-KEY'
AWS_DEFAULT_REGION=us-east-1
```