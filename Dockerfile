# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the Pipfile and Pipfile.lock to the working directory
COPY Pipfile Pipfile.lock /app/

# Install pipenv
RUN pip install pipenv

# Install the dependencies using pipenv
RUN pipenv install --deploy --system

# Set Terraform version
ARG TERRAFORM_VERSION=1.9.3

# Install Terraform, AWS CLI, and Docker CLI
RUN apt-get update && \
    apt-get install -y wget unzip curl gnupg2 lsb-release && \
    wget https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip && \
    unzip terraform_${TERRAFORM_VERSION}_linux_amd64.zip && \
    mv terraform /usr/local/bin/ && \
    rm terraform_${TERRAFORM_VERSION}_linux_amd64.zip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    echo "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    apt-get clean

# Copy the current directory contents into the container at /app
COPY . /app

# Make the start script executable
RUN chmod +x /app/start.sh

# Create the .kaggle directory and copy kaggle.json into it
RUN mkdir -p /root/.kaggle && cp /app/kaggle.json /root/.kaggle/kaggle.json

# Ensure the kaggle.json file has the correct permissions
RUN chmod 600 /root/.kaggle/kaggle.json

# Copy the config directory
COPY config /app/config

# Copy the dashboards directory
COPY dashboards /app/dashboards

# Set the working directory to /app/src
WORKDIR /app/src

# Expose the necessary ports
EXPOSE 5000 8080

CMD ["/bin/bash"]
