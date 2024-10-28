#!/bin/bash
# Load environment variables
set -a  # automatically export all variables
source /app/.env
set +a

# Navigate to the Terraform directory
cd /app/infrastructure/

# Initialize and reconfigure Terraform
terraform init -reconfigure
terraform apply -auto-approve

# Navigate back to the main directory
cd /app

# Log in to AWS ECR
aws ecr get-login-password --region $TF_VAR_aws_region | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$TF_VAR_aws_region.amazonaws.com

# Tag the current Docker image
docker tag $TF_VAR_ecr_repository_name:latest $AWS_ACCOUNT_ID.dkr.ecr.$TF_VAR_aws_region.amazonaws.com/$TF_VAR_ecr_repository_name:latest

# Push Docker image to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$TF_VAR_aws_region.amazonaws.com/$TF_VAR_ecr_repository_name:latest
