#!/bin/bash

export CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1

clearml-agent daemon \
    --queue pipeline-controller \
    --docker bottles-pipeline:latest