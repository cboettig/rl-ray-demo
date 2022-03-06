#!/bin/bash

rllib train  --checkpoint-at-end -f fishing-PPO-config.yml

rllib evaluate saved_checkpoint/checkpoint/checkpoint --episodes 10  --run PPO 
