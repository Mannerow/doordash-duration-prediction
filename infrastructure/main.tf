# Make sure to create state bucket beforehand
terraform {
  required_version = ">= 1.0"
  backend "s3" {
    bucket  = var.state_bucket
    key     = "mlops-zoomcamp-stg.tfstate"
    region  = var.aws_region
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
