import torch
import gym
import gym_conservation
import gym_fishing
import os
from ray import tune
from ray.rllib import agents
from ray.rllib.agents.ppo import PPOTrainer
torch.cuda.device_count()


## rllib ignores gym registered names, need to register the ones we want manually.
## We load each with it's default configuration here
## note these envs were not written to take a single parameter dictionary ("config")
tune.register_env("conservation-v6", lambda config: gym_conservation.envs.NonStationaryV6())
tune.register_env("conservation-v5", lambda config: gym_conservation.envs.NonStationaryV5())
tune.register_env("fishing-v0", lambda config: gym_fishing.envs.FishingEnv())
tune.register_env("fishing-v1", lambda config: gym_fishing.envs.FishingCtsEnv())

os.environ["RLLIB_NUM_GPUS"] = str(torch.cuda.device_count())
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'


# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "fishing-v1",
    # Use 4 parallel environment workers (what SB3 calls "vector environments")
    "num_workers": 4,
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    "num_gpus": 1,
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    "evaluation_num_workers": 1,
    "evaluation_interval": 2,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    }
}

# Create our RLlib Trainer.
trainer = PPOTrainer(config=config)


## Here we go, this is the slow part.  Will automatically log to Tensorboard (~/ray_results)
for _ in range(1000):
    trainer.train()

checkpoint = trainer.save()

# Evaluate a saved agent:

model = PPOTrainer(config=config)
model.load_checkpoint(checkpoint)

# Evaluate the trained Trainer (and render each timestep to the shell's output).
eval = trainer.evaluate()
print(eval)
