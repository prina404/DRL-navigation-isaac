USE_PIP_ISAAC=true

if [ -z ${ISAACSIM_PATH+x} ]; then
    echo "ISAACSIM_PATH is not set, assuming IsaacSim is not pre-installed."
else
    ISAACSIM_VERSION=$(cat ${ISAACSIM_PATH}/VERSION | awk -F='-' '{print $1}')
    if [ "$ISAACSIM_VERSION" != "5.1.0" ]; then
        echo "Detected binary Isaacsim version different from 5.1.0. Pip's version of IsaacSim will be installed instead."
    fi
fi


LDD_VER=$(ldd --version | awk '/ldd/{print $NF}')
echo "Detected ldd version: $LDD_VER"
# check ldd version is greater than 2.35
if [ "$USE_PIP_ISAAC" = true ] && ! printf '%s\n' "2.35" "$LDD_VER" | sort -VC; then
    echo "ldd version is less than 2.35, which may cause issues with pip's version of IsaacSim. Please install IsaacSim using pre-built binaries."
    exit 1
fi

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
git clone --branch v2.3.2 git@github.com:isaac-sim/IsaacLab.git

if [ "$USE_PIP_ISAAC" = true ]; then
    pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
else
    ln -s ${ISAACSIM_PATH} IsaacLab/_isaac_sim
fi

./IsaacLab/isaaclab.sh -i
pip install -e .
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# ViNT repo setup
git clone https://github.com/robodhruv/visualnav-transformer.git
cd visualnav-transformer/train
pip install -e .




