#!/bin/bash

rllib train -f fishing-PPO-basic-config.yml
rllib train -f fishing-ARS-config.yml

## check scores 
## does not understand tilde expansi
CHECKPOINT="/home/cboettig/ray_results/fishing-v2-ppo/PPO_gym_fishing.envs.FishingModelError_d310e_00000_0_2022-09-18_22-32-49/checkpoint_002500"
rllib evaluate $CHECKPOINT --episodes 10  --run PPO 
