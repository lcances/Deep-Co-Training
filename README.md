# UrbanSound8K
UrbanSound8K experimentation

dataset:

`http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf`

# Required package
```bash
conda create -n ubS8k python=3 pip
conda activate Ubs8k
conda install pytorch
conda install pandas
conda install numpy
conda install h5py
conda install pillow
conda install scikit-image
pip install advertorch
pip install tensorboard
pip install librosa
pip install tqdm
```

# Prepare the dataset
The system make use of HDF file to greatly reduce loading time.
- **-l** set the cropping & padding size of each file
- **-sr** set the sampling rate used to load the audio
- **-a** path to the audio (hdf file will be save here)
```bash
conda activate ubS8k
cd standalone
python mv_to_hdf.py -sr 22050 -l 4 -a <path/to/audio/directory>
```

# Some standalone scripts
### Full supervised with and without augmentation
##### Simple run, default parameters
```bash
conda activate ubS8k
cd standalone
python full_supervised.py -t 1 2 3 4 5 6 7 8 9 -v 10 -T test
```

Detailed run:
- model should be available under the form of a class or a function in *models.py*
- augmentation available are the one describe in:
    - signal_augmentations.py
    - spec_augmentations.py
    - img_augmentations.py
```bash
conda activate ubs8k
cd standalone
python full_supervised_aug.py \
    -t 1 2 3 4 5 6 7 8 9 \                                # training folds
    -v 10 \                                               # validation fold(s)
    --subsampling 0.1 \                                   # use only 10 % of the dataset
    --subsampling_method balance \                        # pick sampling fairly among each class
    --model scallable2 \                                  # use model call scallable2
    -a="signal_augmentations.Noise(0.5, target_snr=15)" \ # augmentation to apply for training
    --num_workers 8 \                                     # use 8 process for training
    --log info \                                          # display log of level INFO and above
    -T full_supervised_example                            # tensorboard directory output
```

##### Grid search
The script *script_augmentation.py* perform a grid search by applying unique augmentation and
train a model with **-t 1 2 3 4 5 6 7 8 9** and **-v 10**.

The --job_name parameters is automatically fill with the augmentation name
```bash
conda activate ubs8k
cd standalone
python script_full_supervised_crossval.py \
    --subsampling 0.1 \                                   # use only 10 % of the dataset
    --subsampling_method balance \                        # pick sampling fairly among each class
    --model scallable2 \                                  # use model call scallable2
    --num_workers 8 \                                     # use 8 process for training
    --log info \                                          # display log of level INFO and above
    -T GS_unique_augmentation                             # tensorboard directory output
```


### Co-Training with and without augmentation
Simple run, default parameters
```bash
conda activate ubs8k
cd standalone
python co-training.py -t 1 2 3 4 5 6 7 8 9 -v 10 -T test
```

Detailed run:
- model should be available under the form of a class or a function in *models.py*
- augmentation available are the one describe in:
    - signal_augmentations.py
    - spec_augmentations.py
    - img_augmentations.py
```bash
conda activate ubs8k
cd standalone
python co-training.py \
    -t 1 2 3 4 5 6 7 8 9 \
    -v 10
    --subsampling 0.1 \                                   # use 10% of the dataset
    --subsampling_method balance \                        # pick sample fairly among each class
    --model scallable2 \                                  # model to use
    --nb_view 2 \                                         # nb view for co-training (must be multiple of 2)
    --ratio 0.1 \                                         # amount of supervised file to use
    --batchsize 100 \                                     
    --lambda_cot_max 10 \                                 # co-training variable
    --lambda_diff_max 0.5 \                               # co-training variable
    --epsilon 0.02 \                                      # epsilon for adversarial generation
    --warm_up 80 \                                        # warmup length for concerned variables
    --base_lr 0.05 \                                      # initial learning rate
    --decay 0.001 \                                       # weight decay for optimizer (SGD)
    --momentum 0.0 \                                      # momentum for optimizer (SGD)
    -a="signal_augmentations.Noise(0.5, target_snr=15)" \ # augmentation to apply for training
    --num_workers 8 \                                     # use 8 process for training
    --log info \                                          # display log of level INFO and above
    -T co-training_example                                # tensorboard directory output
```
