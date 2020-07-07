# UrbanSound8K

Application of Deep Co-Training for audio tagging on the UrbanSound8K dataset.

# Requirements
```bash
conda create -n dct python=3 pip
conda activate ubs8k
conda install pytorch
conda install pandas
conda install numpy
conda install h5py
conda install pillow
conda install tqdm
conda install -c conda-forge librosa
pip install advertorch
pip install tensorboard
pip install scikit-image

# if not automatically install
pip install torchvision # dependency for advertorch

pip install --upgrade git+https://github.com/leocances/pytorch_metrics.git@v2 # <-- personnal pytorch metrics functions
pip install --upgrade git+https://github.com/leocances/augmentation_utils.git # <-- personnal audio & image augmentation 
pip install --upgrade git+https://github.com/leocances/UrbanSound8K.git # UBS8K dataset manager
```



# Prepare the dataset

- Download the dataset: `http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf`

- Convert to HDF File:
    
    The system make use of HDF file to greatly reduce loading time.
    - **-l** set the cropping & padding size of each file
    - **-sr** set the sampling rate used to load the audio
    - **-a** path to the audio (hdf file will be save here)
```bash
conda activate ubs8k
cd UrbanSound8k/standalone
python mv_to_hdf.py -sr 22050 -l 4 -a <path/to/audio/directory>
```

# Best results and reproducibility
|System                         | Accuracy ± std |
|---------------------------    |----------------|
|Supervised                     | 47.3 ± 4.1     |
|Deep Co-Training               | 55.4 ± 4.6     |
|**Augmented Deep Co-Training** | **59.7 ± 5.1** |

### Dynamic augmentation
Compute dynamically the augmentation when needed. The augmentation are compute on CPU and it is fairly long to do so,
hence the high number of workers. (*--num_workers 16*)
```bash
conda activate ubs8k

FOLDS=(
	"-t 2 3 4 5 6 7 8 9 10 -v 1" \
	"-t 1 3 4 5 6 7 8 9 10 -v 2" \
	"-t 1 2 4 5 6 7 8 9 10 -v 3" \
	"-t 1 2 3 5 6 7 8 9 10 -v 4" \
	"-t 1 2 3 4 6 7 8 9 10 -v 5" \
	"-t 1 2 3 4 5 7 8 9 10 -v 6" \
	"-t 1 2 3 4 5 6 8 9 10 -v 7" \
	"-t 1 2 3 4 5 6 7 9 10 -v 8" \
	"-t 1 2 3 4 5 6 7 8 10 -v 9" \
	"-t 1 2 3 4 5 6 7 8 9 -v 10" \
)

for fold_idx in ${!folds[*]}
do
    job_name="final_run${fold_idx}"
    python UrbanSound8k/standalone/co-training.py \
        ${folds[$fold_idx]} \
        --job_name ${job_name} \
        --model cnn \
        --base_lr 0.01 \
        --lambda_cot_max 5 \
        --lambda_diff_max 0.25 \
        --warm_up 160 \
        --epsilon 0.1 \
        --epochs 400 \
        --num_workers 16 \  # Another varient exist for computer with low CPU count
        --augment_S \
        -a="signal_augmentations.PitchShiftChoice(0.75, choice=(-3, -2, 2, 3))"
done
```

### Static augmentation
If you don't possess a computer with a lot of cpu core, then you might be interested by using pre-computed augmentations.

**Pre-compute augmentation**
```bash
for i in 1 2 3 4 5 6 # we want to create 6 flavors of the same augmentation 
do
    python UrbanSound8k/standalone/preprocess_augmentation.py \
        --audio_root dataset/audio \
        --sampling_rate 22050 \
        --length 4 \
        --num_workers 4 \
        -A="signal_augmentations.PitchShiftChoice(1.0, choice=(-3, -2, 2, 3))"
done
```

**Reproduce results using pre-computed augmentation**
```bash
conda activate ubs8k

FOLDS=(
	"-t 2 3 4 5 6 7 8 9 10 -v 1" \
	"-t 1 3 4 5 6 7 8 9 10 -v 2" \
	"-t 1 2 4 5 6 7 8 9 10 -v 3" \
	"-t 1 2 3 5 6 7 8 9 10 -v 4" \
	"-t 1 2 3 4 6 7 8 9 10 -v 5" \
	"-t 1 2 3 4 5 7 8 9 10 -v 6" \
	"-t 1 2 3 4 5 6 8 9 10 -v 7" \
	"-t 1 2 3 4 5 6 7 9 10 -v 8" \
	"-t 1 2 3 4 5 6 7 8 10 -v 9" \
	"-t 1 2 3 4 5 6 7 8 9 -v 10" \
)

for fold_idx in ${!folds[*]}
do
    job_name="final_run${fold_idx}"
    python UrbanSound8k/standalone/co-training_static_aug.py \
        ${folds[$fold_idx]} \
        --job_name ${job_name} \
        --model cnn \
        --base_lr 0.01 \
        --lambda_cot_max 5 \
        --lambda_diff_max 0.25 \
        --warm_up 160 \
        --epsilon 0.1 \
        --epochs 400 \
        --num_workers 16 \  # Another varient exist for computer with low CPU count
        --augment_S \
        --static_augments="{(PSC1': 0.75}"
done
```

<!-- 
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

# Reproductibility
### Best model
```bash
conda activate ubs8k
cd standalone
python co-training.py --model scallable2 --base_lr 0.01 --lambda_cot_max 2 --lambda_diff_max 0.5 --warm_up 120 --epsilon 0.02 --parser_ratio 0.40 --num_workers 20 --epochs 400 --tensorboard_dir moreS_PSC1_0.75_full --log info --augment_S -a="signal_augmentations.PitchShiftChoice(0.75, choice=(-3, -2, 2, 3))"

## For my personnal use
For my personnal usage, workaround on CALMIP (limited user space and hardlink not working between different divice)
 - 5Go is not enough to install everything at once.
 - It need some `conda clean --all` after installing big module (pytorch)
 - Best to have miniconda install in tmpdir directory
 - If not, have the venv directory inside the project and create a symlink
 `cd /miniconda/envs; ln -s /path/to/venv/ <name>`
 - Conda doesn't like symlink. use `CONDA_ALWAY_COPY=true` before calling conda
 - Pip cache is store under `~/.cache`
 
```Bash
CONDA_ALWAYS_COPY=true conda create -p /path/to/venv/ python=3 pip
cd ~/miniconda3/envs
ln -s /path/to/venv/ ubs8k
conda activate ubs8k

CONDA_ALWAYS_COPY=true conda install pytorch
conda clean --all
CONDA_ALWAYS_COPY=true conda install pandas numpy
...
```
-->
