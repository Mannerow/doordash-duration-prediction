# Make sure to manually create state bucket beforehand
terraform {
  required_version = ">= 1.0"
  backend "s3" {
    bucket  = "terraform-state-bucket-mm"
    key     = "mlops-zoomcamp-stg.tfstate"
    region  = "us-east-1"
    encrypt = true
  }
}

# Cloud provider
provider "aws" {
  region = var.aws_region
}


data "aws_caller_identity" "current_identity" {}

locals {
  account_id = data.aws_caller_identity.current_identity.account_id
}

module "mlflow_models_bucket" {
  source      = "./modules/s3"
  bucket_name = var.mlflow_models_bucket
}

module "predictions_data_bucket" {
  source      = "./modules/s3"
  bucket_name = var.prediction_bucket
}

module "ecr_repository" {
  source = "./modules/ecr"
  repository_name = var.ecr_repository_name
}
