# rl-ray-demo

## Installation

Many ways of setting up a python environment.  A local venv from scratch is often the most reliable:

```
python -m venv venv
source venv/bin/activate
pip install ray[rllib] ray[tune] tensorboard
pip install git+https://github.com/boettiger-lab/gym_fishing
```

## Getting started



## Tensorboard

simlink `~/ray_results` to `/var/log/tensorboard/<username>`, e.g.

```bash
ln -s ~/ray_results/ /var/log/tensorboard/cboettig
```

(on containers not hosting the tensorboard, simlink will not be read)

**Admin**

(Docker host administrator only, e.g. see <https://github.com/boettiger-lab/servers> for more details)

- Make sure tensorboard is running on the server. (Usually in the primary `rstudio` container).

```bash
tensorboard --logdir /var/log/tensorboard --bind_all --port 2223
```

- Make sure Caddyfile is exposing tensorboard URL (e.g. <https://tensorboard.cirrus.carlboettiger.info>)
- above can be moved into `/etc/services.d/tensorboard/run` as:


```
#!/usr/bin/with-contenv bash
# place this file in /etc/services.d/tensorboard/run   
tensorboard --logdir /var/log/tensorboard/ --bind_all --port 2223 
``` 


## GPU

Monitor GPU use locally with `nvitop`, installable with pip.  
(Or go old-school `watch -n 3 nvidia-smi`).  

If you see `Failed to initialize NVML: Driver/library version mismatch`, container probably needs to be restarted (e.g. after driver updates on the host machine.)

**Admin**: If mismatch occurs on host, stop all tasks using the GPU (`gdm`, monitors, containers, etc) and recursively unload modules, usually:

```bash
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia_uvm
sudo rmmod nvidia
```

Alternatively, just reboot the machine.

If GPU is being lost from container without update, make sure devices are hard-wired in docker runtime:

```bash
docker run ...
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
  --device /dev/nvidia0:/dev/nvidia0
```

