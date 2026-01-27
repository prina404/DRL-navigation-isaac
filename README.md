# isaac-navigation-trainer

## IsaacLab installation instructions
```bash
$ pip install -e .
$ pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
$ git clone --branch main git@github.com:isaac-sim/IsaacLab.git
$ cd IsaacLab && ./isaaclab.sh -i
```

## InteriorAgent dataset 

Install git-xet
```bash
$ curl -sSfL https://hf.co/git-xet/install.sh | sh
```

Clone huggingface repo (25GB of disk required)
```bash
$ git clone https://huggingface.co/datasets/spatialverse/InteriorAgent
```

Setup the `INTERIOR_AGENT_DIR` parameter in `src/cfg/CFG.py` to the corresponding dataset folder