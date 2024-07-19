from prefect import task, flow
import subprocess

@task
def run_train():
    result = subprocess.run(['python', 'train.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"train.py failed: {result.stderr}")
    print(result.stdout)
    return result.stdout  # Return value to be used as input in dependent tasks if needed

@task
def run_hpo():
    result = subprocess.run(['python', 'hpo.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"hpo.py failed: {result.stderr}")
    print(result.stdout)
    return result.stdout  # Return value to be used as input in dependent tasks if needed

@task
def run_register_model():
    result = subprocess.run(['python', 'register_model.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"register_model.py failed: {result.stderr}")
    print(result.stdout)
    return result.stdout  # Return value to be used as input in dependent tasks if needed

@flow
def ml_workflow():
    train_result = run_train()
    hpo_result = run_hpo(wait_for=[train_result])  # Dependency managed by wait_for
    register_model_result = run_register_model(wait_for=[hpo_result])  # Dependency managed by wait_for

# Run the flow
if __name__ == "__main__":
    ml_workflow()
