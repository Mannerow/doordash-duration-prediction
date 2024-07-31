#!/bin/bash
# Load environment variables
set -a  # automatically export all variables
source .env
set +a

# Navigate to the Terraform directory
cd infrastructure/

# Initialize and apply Terraform
terraform init
terraform apply -auto-approve

# Navigate back to the main directory
cd ..

# Build Docker image
docker build -t $TF_VAR_ecr_repository_name .

# Log in to AWS ECR
aws ecr get-login-password --region $TF_VAR_aws_region | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$TF_VAR_aws_region.amazonaws.com

# Tag Docker image
docker tag $TF_VAR_ecr_repository_name:latest $AWS_ACCOUNT_ID.dkr.ecr.$TF_VAR_aws_region.amazonaws.com/$TF_VAR_ecr_repository_name:latest

# Push Docker image to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$TF_VAR_aws_region.amazonaws.com/$TF_VAR_ecr_repository_name:latest

# Start Docker containers
docker compose up