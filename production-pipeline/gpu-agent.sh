export CLEARML_CONFIG_FILE=~/clearml-personal.conf 
export CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1
export CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1
export CLEARML_AGENT_EXTRA_DOCKER_ARGS="--shm-size=6g"
export CLEARML_WORKER_NAME="gpu-agent"

clearml-agent daemon \
    --queue my_laptop-gpu-tasks \
    --docker bottles-pipeline:latest \
    --gpus all