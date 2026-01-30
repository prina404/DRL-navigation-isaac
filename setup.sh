CONDA_ENV_NAME="lfp"

eval "$(~/miniconda3/bin/conda shell.bash hook)"

conda update -n base -c conda-forge conda -y

conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba

conda create -n ${CONDA_ENV_NAME} python=3.11 -y

conda activate ${CONDA_ENV_NAME}

conda install pip git git-lfs cmake -y
conda install conda-forge::hf-xet -y
git xet install
git lfs install
pip install uv

conda env config vars set CUDA_HOME=""
conda activate ${CONDA_ENV_NAME}
conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit -y
conda activate ${CONDA_ENV_NAME}

conda install -c conda-forge cudnn -y
conda install -c conda-forge vulkan-tools -y

conda install conda-forge::libglvnd-egl-cos7-x86_64 -y
conda install conda-forge::egl-probe -y

uv pip install --no-build-isolation -e .
uv pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
git clone --branch main git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh -i
cd ..



