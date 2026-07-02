#!/bin/bash
export CLEARML_CONFIG_FILE=~/clearml-personal.conf 
export CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1
export CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1
export CLEARML_WORKER_NAME="controller-agent"

clearml-agent daemon \
    --queue pipeline-controller \
    --docker bottles-pipeline:latest