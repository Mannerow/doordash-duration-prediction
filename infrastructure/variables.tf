variable "aws_region" {
  description = "AWS region to create resources"
  default     = "us-east-1"
}

variable "mlflow_models_bucket" {
  description = "The name of the S3 bucket which stores the best model"
  default     = "mlflow-models-mannerow"
}

variable "prediction_bucket" {
  description = "The name of the S3 bucket for the scored parquet files"
  default     = "doordash-duration-prediction-mannerow"
}

variable "project_id" {
  description = "project_id"
  default = "mlops-zoomcamp"
}

variable "docker_image_local_path" {
  description = ""
  default     = "../"
}

variable "ecr_repository_name" {
  description = "The name of the ECR repository"
  type        = string
}
