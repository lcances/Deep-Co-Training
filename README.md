# UrbanSound8K
UrbanSound8K experimentation

dataset:

`http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf`

# Required package
```bash
conda create -n UbS8k pip
conda install torch
conda install tensorboard
conda install pandas
conda install numpy
pip install librosa
pip install tqdm
```

# Full supervised
```bash
conda activate UbS8k
cd standalone
python python full_supervised.py -t 1 2 3 4 5 6 7 8 9 -v 10 -T test
```