#!/bin/bash 
# session description here
PYTHONPATH=/home/roblab1/PycharmProjects/MultiAgentEnv python main.py \
--experiment_name test \
--robot_type car \
--algorithm ddpg \
--is_train
--max_steps 150 \
--episodes 50 \
--num_agents 5 \
--render \
--seed 12 \
--fd_model model_Tue_Sep__26_3_21_12.pt
