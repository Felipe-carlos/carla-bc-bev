
#!/bin/bash
# export CUDA_VISIBLE_DEVICES=3
screen -L -Logfile screenlog.2 -S  carla_online_train .venv/bin/python train_kde_online.py