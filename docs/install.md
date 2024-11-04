# Step-by-step installation instructions

**1. Create a conda virtual environment and activate it.**
```shell
conda create -n adaptiveocc python=3.7 -y
conda activate adaptiveocc
```

**2. Install PyTorch and torchvision.**
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

**3. Install mmcv, mmdet, and mmseg.**
```shell
pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
```

**4. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```

**5. Install other dependencies.**
```shell
pip install timm
pip install open3d-python
```

**6. Prepare pretrained models.**
```shell
cd AdaptiveOcc 
mkdir ckpts
cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```
