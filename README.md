# UrbanSound8K
Deep Co-Training experimentation

datasets used:
 - [UrbanSound8k](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf)
 - [DESED](https://project.inria.fr/desed/)
 - [ESC50](https://github.com/karolpiczak/ESC-50)


# Required package
```bash
conda create -n ubS8k python=3 pip
conda activate Ubs8k
conda install pytorch
conda install pandas
conda install numpy
conda install h5py
conda install pillow
pip install advertorch
pip install torchvision
pip install tensorboard
pip install librosa
pip install tqdm
pip install scikit-image

# if not automatically install
pip install torchvision # dependency for advertorch
```

# Require dataset related package
I'm using my own implementation for the management of the different dataset.
 - UrbanSound8K see: `https://github.com/leocances/UrbanSound8K.git`
 - DESED see: `https://github.com/leocances/dcase2020.git`
 - ESC50: `https://github.com/leocances/ESC50.git`

