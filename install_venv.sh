#!/usr/bin/env bash

conda create -n ubs8k python=3 pip
conda activate ubs8k
conda install pytorch
conda install pandas
conda install numpy
conda install h5py
conda install pillow
pip install advertorch
pip install tensorboard
pip install librosa
pip install tqdm
pip install scikit-image

# if not automatically install
pip install torchvision # dependency for advertorch

pip install git+https://github.com/leocances/pytorch_metrics.git # <-- personnal pytorch metrics functions
pip install git+https://github.com/leocances/augmentation_utils.git # <-- personnal audio & image augmentation
pip install git+https://github.com/leocances/UrbanSound8K.git # UBS8K dataset manager
