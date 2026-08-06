sudo apt install python3.12-dev libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev

uv venv --python 3.12 --seed env_isaaclab
source env_isaaclab/bin/activate
uv pip install --upgrade pip

uv pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
uv pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

git clone git@github.com:isaac-sim/IsaacLab.git --branch release/3.0.0-beta2
./IsaacLab/isaaclab.sh -i

uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
uv pip install -e .