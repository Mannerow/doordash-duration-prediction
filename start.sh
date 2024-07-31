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

# Start Docker containers
docker compose up