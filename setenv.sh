#!/bin/bash
pip install ninja
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install "git+https://github.com/NVlabs/nvdiffrast.git"
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
# conda env update -f env.yaml
pip install diffusers==0.20.2   # for 0123Material
pip install huggingface_hub==0.24.7