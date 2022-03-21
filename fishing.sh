#!/bin/bash

rllib train -f fishing-PPO-config.yml
rllib train -f fishing-ARS-config.yml

rllib evaluate saved_checkpoint/checkpoint/checkpoint --episodes 10  --run PPO 
