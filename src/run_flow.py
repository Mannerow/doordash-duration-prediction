from prefect import task, flow
import subprocess
from prefect import serve
from datetime import timedelta

@task
def run_data_preprocess():
    result = subprocess.run(['python', 'src/data_preprocess.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"data_preprocess.py failed: {result.stderr}")
    print(result.stdout)
    return result.stdout  # Return value to be used as input in dependent tasks if needed

@task
def run_train():
    result = subprocess.run(['python', 'src/train.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"train.py failed: {result.stderr}")
    print(result.stdout)
    return result.stdout  # Return value to be used as input in dependent tasks if needed

@task
def run_hpo():
    result = subprocess.run(['python', 'src/hpo.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"hpo.py failed: {result.stderr}")
    print(result.stdout)
    return result.stdout  # Return value to be used as input in dependent tasks if needed

@task
def run_register_model():
    result = subprocess.run(['python', 'src/register_model.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"register_model.py failed: {result.stderr}")
    print(result.stdout)
    return result.stdout  # Return value to be used as input in dependent tasks if needed

@task
def run_score_batch():
    result = subprocess.run(['python', 'src/score_batch.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"score_batch.py failed: {result.stderr}")
    print(result.stdout)
    return result.stdout  # Return value to be used as input in dependent tasks if needed

@flow(log_prints=True)
def ml_workflow():
    data_preprocess_result = run_data_preprocess()
    train_result = run_train(wait_for=[data_preprocess_result])
    hpo_result = run_hpo(wait_for=[train_result])  # Dependency managed by wait_for
    register_model_result = run_register_model(wait_for=[hpo_result])  # Dependency managed by wait_for
    score_batch_result = run_score_batch(wait_for=[register_model_result])  # Dependency managed by wait_for

# Serve the flow with a schedule
if __name__ == "__main__":
    ml_workflow.serve(
        name="ml-workflow-deployment",
        parameters={},
        interval=timedelta(hours=1).total_seconds()  # Set the interval in seconds
    )
