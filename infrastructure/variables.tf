variable "aws_region" {
  description = "AWS region to create resources"
  default     = "us-east-1"
}

variable "state_bucket" {
  description = "The name of the S3 bucket for Terraform state storage"
  default     = "tf-state-mlops-zoomcamp-mm"
}

variable "project_id" {
  description = "project_id"
  default = "mlops-zoomcamp"
}

variable "model_bucket" {
  description = "mlflow-models-mannerow"
}

variable "docker_image_local_path" {
  description = ""
}

variable "ecr_repo_name" {
  description = ""
}