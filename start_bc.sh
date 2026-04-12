
#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/
export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export CUDA_VISIBLE_DEVICES=3
screen -L -Logfile screenlog.2 -S  carla_bc_cvt /home/jovyan/.pyenv/shims/python learn_bc.py