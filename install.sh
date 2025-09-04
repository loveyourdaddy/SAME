#!/bin/bash

### prerequisite
# sudo apt-get install freeglut3-dev

## create and activate custom conda env
# conda update --all
# conda create -n same python=3.8
# conda activate same

## torch 2.3 stable, Cuda 11.8
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install torch_geometric

pip install scipy tqdm IPython tensorboard matplotlib plotly pyyaml
pip install pynvml gputil

pip install glfw
pip install imgui
pip install opencv-python

python setup.py develop

# submodule: fairmotion (https://github.com/sunny-Codes/fairmotion)
git submodule init
git submodule update
cd src/fairmotion
python setup.py develop

