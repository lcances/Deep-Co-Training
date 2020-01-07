# UrbanSound8K
UrbanSound8K experimentation

dataset:

`http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf`

# Required package
```bash
conda create -n UbS8k pip
conda install pytorch
conda install pandas
conda install numpy
pip install tensorboard
pip install librosa
pip install tqdm
```

# Prepare the dataset
The system make use of HDF file to greatly reduce loading time
```bash
conda activate UbS8k
cd standalone
python mv_to_hdf.py -sr 22050 -l 4 -a <path/to/audio/directory>
```

# Full supervised
```bash
conda activate UbS8k
cd standalone
python full_supervised.py -t 1 2 3 4 5 6 7 8 9 -v 10 -T test
```
