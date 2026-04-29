# DRL quadruped hybrid navigation

## Installation
It is recommended to install IsaacSim's 5.1.0 pre-built binaries, following [Nvidia's official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html). If no binary installation is detected, the setup script will use isaacsim pip package, which lacks VsCode IntelliSense support

```bash
$ ./setup.sh
$ source .venv/bin/activate
```

### Dataset setup

Clone huggingface repo (25GB of disk required)
```bash
# setup git-xet
$ curl -sSfL https://hf.co/git-xet/install.sh | sh

$ git clone https://huggingface.co/datasets/spatialverse/InteriorAgent

# preprocess the dataset
$ python3 src/preprocessing/preprocess_interioragent.py 
```

Set the `dataset_folder` parameter in `dataset_cfg.yaml` to the InteriorAgent repo folder.

### Weights & Biases setup (optional)
Add the wandb username and project name into a .env file:
```bash
$ echo -e "WANDB_USERNAME=<user> \nWANDB_PROJECT=<project>" > .env
```


# Usage
Train a simple policy that navigates in empty environments

```bash
$ python3 train_policy.py --num_envs=128 --max_iterations=500 --task="go2_lidar_empty" --headless
```

The available tasks are:

```
- go2_lidar_empty
- go2_lidar_full
- go2_vision_empty
- go2_vision_full
- go2_depth_full
```

where the `*_full` tasks build the entire scene, whereas `*_empty` ones are just an empty plane.

The complete MDP configurations are defined in the `src/tasks/go2_*.py` files.
