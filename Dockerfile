# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Copy the Pipfile and Pipfile.lock to the working directory
COPY Pipfile Pipfile.lock /app/

# Install pipenv
RUN pip install pipenv

# Install the dependencies using pipenv
RUN pipenv install --deploy --system

# Copy the current directory contents into the container at /app
COPY . /app

# Create the .kaggle directory and copy kaggle.json into it
RUN mkdir -p /root/.kaggle && cp /app/kaggle.json /root/.kaggle/kaggle.json

# Ensure the kaggle.json file has the correct permissions
RUN chmod 600 /root/.kaggle/kaggle.json

# Set the working directory to /app/src
WORKDIR /app/src

# Expose the necessary ports
EXPOSE 5000 8080

# Define the command to be run when the container starts
CMD ["/bin/bash"]
