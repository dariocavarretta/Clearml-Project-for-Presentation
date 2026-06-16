export CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1
export CLEARML_AGENT_EXTRA_DOCKER_ARGS="--shm-size=6g"

clearml-agent daemon \
    --queue my_laptop-gpu-tasks \
    --docker bottles-pipeline:latest 