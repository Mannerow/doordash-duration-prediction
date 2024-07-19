from prefect import task, Flow
import subprocess

@task
def run_train():
    result = subprocess.run(['python', 'train.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"train.py failed: {result.stderr}")
    print(result.stdout)

@task
def run_hpo():
    result = subprocess.run(['python', 'hpo.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"hpo.py failed: {result.stderr}")
    print(result.stdout)

@task
def run_register_model():
    result = subprocess.run(['python', 'register_model.py'], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"register_model.py failed: {result.stderr}")
    print(result.stdout)

# Define the flow
flow = Flow("ML Workflow")

# Add tasks to the flow
train_task = run_train()
hpo_task = run_hpo(upstream_tasks=[train_task])
register_model_task = run_register_model(upstream_tasks=[hpo_task])

flow.add_task(train_task)
flow.add_task(hpo_task)
flow.add_task(register_model_task)

# Run the flow
if __name__ == "__main__":
    flow.run()
