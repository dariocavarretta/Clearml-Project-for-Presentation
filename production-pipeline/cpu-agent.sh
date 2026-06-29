export CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1
export CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1

clearml-agent daemon \
    --queue my_laptop-cpu-tasks \
    --docker bottles-pipeline:latest \
    