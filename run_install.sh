# cuda 11.6 python 3.8.5
conda create -n avatarpose python=3.8.5
conda activate avatarpose
pip install torch==1.13.1+cu116 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/NVlabs/tiny-cuda-nn/@v1.6#subdirectory=bindings/torch
pip install fvcore iopath
pip install pytorch-lightning==1.5.7
pip install opencv-python
pip install imageio
pip install smplx==0.1.28
pip install hydra-core==1.1.2
pip install h5py ninja chumpy numpy==1.22.4
pip install lpips
pip install nvidia-ml-py3
pip install aitviewer==1.9.0
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.1_cu116.html
pip install open3d==0.10.0