# doordash-duration-prediction

## 📄 Problem Description

The objective of this project is to predict the delivery duration for DoorDash orders using historical data. This involves developing a machine learning model that can accurately forecast the time it will take for a delivery to reach the customer from the moment an order is placed. The project will leverage MLOps principles to ensure robust experimentation, model tracking, monitoring, infrastructure provisioning, and automation throughout the development and deployment lifecycle.

## 📊 Dataset

The dataset used for this project is sourced from Kaggle and contains various features related to DoorDash orders, such as pickup and drop-off locations, timestamps, order size, and other relevant attributes. The dataset can be found at DoorDash ETA Prediction Dataset.

## 🔒 Authentication

This project requires Kaggle, AWS, and Prefect accounts. Kaggle is used to download the dataset, AWS is used to store the model and predictions in S3 and the Docker images with ECR, and Prefect is used for pipeline orchestration.

## ⚙️ Installation

To get started with the project, follow these steps:

**1. Install Terraform:**

Visit this [link](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) and follow the instructions to install Terraform on your system.

**2. Clone the repository:**

```bash
git clone https://github.com/Mannerow/doordash-duration-prediction
cd doordash-duration-prediction
```

**3. Set up Authentication:**

1. Place your kaggle.json in the root directory of this project.
2. Create a .env file which will contain the necessary environment variables for AWS and Prefect authentication, S3 bucket and ECR names, and the MLFLow Endpoint. 

```bash
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_DEFAULT_ARTIFACT_ROOT='S3-BUCKET-PATH'
MLFLOW_BACKEND_STORE_URI='sqlite:///backend.db'
AWS_ACCESS_KEY_ID='YOUR-ACCESS-KEY'
AWS_SECRET_ACCESS_KEY='YOUR-SECRET-ACCESS-KEY'
AWS_DEFAULT_REGION=us-east-1
AWS_ACCOUNT_ID='YOUR-ACCOUNT-ID'
PREFECT_API_KEY='YOUR-PREFECT-API-KEY'
PREFECT_WORKSPACE='YOUR-PREFECT-WORKSPACE-NAME'
TF_VAR_aws_region=us-east-1
TF_VAR_mlflow_models_bucket='MODEL-BUCKET-NAME'
TF_VAR_prediction_bucket='PREDICTION-BUCKET-NAME'
TF_VAR_ecr_repository_name='ECR-REPO-NAME'
```

**4. Manually create an S3 bucket for the Terraform state:**

Create an S3 bucket and update the `'/infrastructure.main.tf'` file to reflect the new bucket. 

**5. Additional Step for Windows Users:**

Windows users must convert the line endings of their script and environment files to Unix-style. This is because Windows uses a different line ending format (CRLF) compared to Unix/Linux (LF), which can cause issues when running scripts in a Unix-based environment like Docker.

Run the following command to convert the line endings:

```bash
dos2unix start.sh
dos2unix .env
```

## 🚀 Running the Project

To run the project, simply run the following command. Docker will create services for `mlflow`, `postgres`, `adminer`, `grafana`, and `app`. The `app` service will execute a start script that initializes and applies `Terraform` before running the flow.

```bash
docker compose up
```

## 🌐 Viewing Prefect Cloud

The app service provides a link [https://app.prefect.cloud/auth/resume](https://app.prefect.cloud/auth/resume) to Prefect Cloud. Open this link to access the Prefect Cloud UI, which offers detailed insights into deployments and previous runs. While the deployment is scheduled to run hourly, you can manually initiate a run by clicking 'Run' for immediate execution.

## 🛠️ Infrastructure Provisioning with Terraform

This project leverages Terraform to automate the setup of AWS S3 buckets for storing model artifacts and predictions, as well as an Amazon ECR repository for the Docker image. Utilizing Terraform ensures that the infrastructure is provisioned consistently and reproducibly, reducing the risk of manual errors. This automation is crucial for maintaining a reliable workflow, especially in a production environment. It is important to note that the user should manually create an S3 bucket for storing the Terraform state, which helps in tracking infrastructure changes over time and enables collaborative infrastructure management.

## 🧪 Experiment and Model Tracking with MLFlow

This project utilizes MLFlow for experiment and model tracking, allowing users to log metrics, parameters, and artifacts for their machine learning experiments. By providing a centralized place to track model performance, MLFlow ensures reproducibility and simplifies the comparison of different model runs. This tool is essential for maintaining an organized workflow and improving model management over time. To view and manage experiments, navigate to [http://localhost:5000/](http://localhost:5000/).

## 📈 Monitoring with Evidently and Grafana

For effective monitoring of the model's performance and data quality, this project utilizes Evidently and Grafana. Users can access the monitoring dashboard by navigating to [http://localhost:3000/login](http://localhost:3000/login) and logging in with the username 'admin' and password 'admin'. Once logged in, navigate to the dashboards section to view a comprehensive dashboard that displays key metrics such as test RMSE, prediction drift, the number of drifted columns, and the number of missing values. This setup ensures continuous insight into the model's performance and data integrity, facilitating prompt detection and resolution of any issues.

## 🔍 How It Works

This section provides an overview of the primary scripts and their functionalities within the project, illustrating the workflow from data preprocessing to model deployment and monitoring:

**`experimentation.ipynb`**

This Jupyter notebook is used for exploratory data analysis (EDA) and initial experimentation with the DoorDash delivery duration prediction dataset. It imports necessary packages and downloads the dataset from Kaggle. The notebook includes steps for data exploration, such as loading the dataset into a Pandas DataFrame, checking for missing values, and creating new features like delivery duration. It also performs data visualization to identify outliers and understand data distributions. The notebook demonstrates the process of data cleaning by removing outliers and preparing features for modeling. It concludes with building a simple linear regression model using a pipeline, splitting the data into training and test sets, and evaluating model performance using metrics like RMSE and MAE. This notebook serves as an interactive environment to explore and experiment with different approaches before formalizing them into the main project pipeline.

**`start.sh`**

This Bash script automates the setup and deployment process for the project. It begins by loading environment variables from a `.env file` and exporting them. The script then navigates to the Terraform directory to initialize and apply the Terraform configurations, setting up the necessary infrastructure. After returning to the main directory, it logs in to AWS ECR. The script tags the Docker image and pushes it to the ECR repository.

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

This script utilizes multiple libraries such as `pandas`, `boto3`, `joblib`, `psycopg`, `scipy`, `mlflow`, and `evidently` to monitor and store metrics in PostgreSQL. It begins by initializing MLflow for model tracking and loading the best model from an S3 bucket. The script then sets up the monitoring process by loading and predicting on test data, decoding dataframes, and defining column mappings for numerical and categorical features. It uses the Evidently library to generate a report that includes metrics like column drift, dataset drift, missing values, and regression quality. These metrics are then calculated and stored in a PostgreSQL database on a daily basis, filtered by the 'created_at' timestamp, with each day's data processed individually. The reports can be viewed and visualizations can be created by logging into the Grafana interface.

**`utils.py`**

This script contains utility functions that are utilized across different project scripts, promoting modularity and reusability within the codebase. Specifically, it includes helper functions for loading and dumping pickle files, and a function to decode a one-hot encoded DataFrame back to its original format using the DictVectorizer.

## 🔄 Reproducability

To ensure reproducibility, this project’s environment is managed with `pipenv`. While `pipenv` is the primary tool for handling dependencies, a `requirements.txt` file is also included for reference purposes. This helps users understand the specific packages and versions used in the project.