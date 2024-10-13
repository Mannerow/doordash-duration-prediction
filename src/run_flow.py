import subprocess
from datetime import timedelta

from prefect import flow, serve, task


@task
def run_data_preprocess():
    result = subprocess.run(
        ["python", "data_preprocess.py"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"data_preprocess.py failed: {result.stderr}")
    print(result.stdout)
    return (
        result.stdout
    )  # Return value to be used as input in dependent tasks if needed


@task
def run_train():
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"train.py failed: {result.stderr}")
    print(result.stdout)
    return (
        result.stdout
    )  # Return value to be used as input in dependent tasks if needed


@task
def run_hpo():
    result = subprocess.run(["python", "hpo.py"], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"hpo.py failed: {result.stderr}")
    print(result.stdout)
    return (
        result.stdout
    )  # Return value to be used as input in dependent tasks if needed


@task
def run_register_model():
    result = subprocess.run(
        ["python", "register_model.py"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"register_model.py failed: {result.stderr}")
    print(result.stdout)
    return (
        result.stdout
    )  # Return value to be used as input in dependent tasks if needed


@task
def run_score_batch():
    result = subprocess.run(
        ["python", "score_batch.py"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"score_batch.py failed: {result.stderr}")
    print(result.stdout)
    return (
        result.stdout
    )  # Return value to be used as input in dependent tasks if needed


@task
def run_monitor_metrics():
    result = subprocess.run(
        ["python", "monitor_metrics.py"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"monitor_metrics.py failed: {result.stderr}")
    print(result.stdout)
    return (
        result.stdout
    )  # Return value to be used as input in dependent tasks if needed


@flow(log_prints=True)
def ml_workflow():
    print("🔄 Preprocessing the data...")

    data_preprocess_result = run_data_preprocess()

    print("🏋️ Training the models...")

    train_result = run_train(wait_for=[data_preprocess_result])

    print("🎛️ Tuning hyperparameters...")

    hpo_result = run_hpo(wait_for=[train_result])  # Dependency managed by wait_for

    print("🏆 Registering the best model...")

    register_model_result = run_register_model(
        wait_for=[hpo_result]
    )  # Dependency managed by wait_for

    print("🔮 Making predictions...")

    score_batch_result = run_score_batch(
        wait_for=[register_model_result]
    )  # Dependency managed by wait_for

    print("📊 Monitoring...")

    monitor_metrics_result = run_monitor_metrics(
        wait_for=[score_batch_result]
    )  # Dependency managed by wait_for


# Serve the flow with a schedule
if __name__ == "__main__":
    # Serve the flow with a schedule, running every 10 minutes
    print("Serving the flow with a schedule...")
    ml_workflow.serve(
        name="ml-workflow-deployment",
        parameters={},
        interval=timedelta(minutes=10).total_seconds(),  # Run every 10 minutes
    )
