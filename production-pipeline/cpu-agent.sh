export CLEARML_CONFIG_FILE=~/clearml-personal.conf 
export CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1
export CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1
export CLEARML_WORKER_NAME="cpu-agent"

clearml-agent daemon \
    --queue my_laptop-cpu-tasks \
    --docker bottles-pipeline:latest \
    