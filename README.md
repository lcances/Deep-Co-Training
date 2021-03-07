# COMPARISON OF DEEP CO-TRAINING AND MEAN-TEACHER APPROACHES FOR SEMI-SUPERVISED AUDIO TAGGING
## Official implementation

Recently, a number of semi-supervised learning (SSL) methods, in the framework of deep learning (DL), were shown to achieve state-of-the-art results on image datasets, while using a (very) limited amount of labeled data. To our knowledge, these approaches adapted and applied to audio data are still sparse, in particular for audio tagging (AT). In this work, we adapted the Deep-Co-Training algorithm (DCT) to perform AT, and compared it to another SSL approach called Mean Teacher (MT), that has been used by the winning participants of the DCASE competitions these last two years. Experiments were performed on three standard audio datasets: Environmental Sound classification (ESC-10), UrbanSound8K, and Google Speech Commands. We show that both DCT and MT achieved performance approaching that of a fully supervised training setting, while using a fraction of the labeled data available, and the remaining data as unlabeled data. In some cases, DCT even reached the best accuracy, for instance, 72.6% using half of the labeled data, compared to 74.4% using all the labeled data. DCT also consistently outperformed MT in almost all configurations. For instance, the most significant relative gains brought by DCT reached 12.2% on ESC-10, compared to 7.6% with MT. Our code is available online

# Requirements
- **automatically**
```bash
pip install -r requirements.txt
```

- **manually**
```bash
conda create -n dct python=3 pip
conda activate dct

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install numpy
conda install pandas
conda install scikit-learn
conda install scikit-image
conda install tqdm
conda install h5py
conda install pillow
conda install librosa -c conda-forge

pip install hydra-core
pip install advertorch
pip install torchsummary
pip install tensorboard

cd Deep-Co-Training
pip install -e .
```

## Fix missing package
- It is very likely that the `ubs8k` will be missing. It a code to manage the UrbanSound8K dataset I wrote almost two years ago before I start using `torchaudio`.
- `pytorch_metrics` is a basic package I wrote to handle many of the metrics I used during my experiments.
- `augmentation_utils` is a package I wrote to test and apply many different augmentation during my experiments.
```bash
pip install --upgrade git+https://github.com/leocances/UrbanSound8K.git@new_data_management
pip install --upgrade git+https://github.com/leocances/pytorch_metrics.git@v2
pip install --upgrade git+https://github.com/leocances/augmentation_utils.git
```
I am planning on release a much cleaner implementation that follow the torchaudio rules.

# Install and prepare datasets
- ESC and SpeechCommand are already available on torchaudio so it is not necessary to download them manually.
- The management of these two datasets can be found in the directory `DCT/dataset/`.
- **UrbanSound8k** need to be download separately (see section *Download UBS8K*)

# Train the systems
The directory `standalone/` contains the different scrpit to execute the different semi-supervised methods and the usual supervised approach. Each approach has it own working directory which contain the python script as well
as a `to_run.sh` and `to_run_gs.sh` for easy execution.

The handling of running arguments is done using [hydra](hydra.cc) and the configuration files can be found in the directory `config/`

## Exemple of supervised cross-validation for ESC-10
```bash
conda activate dct
cd Deep-Co-training/standalone/supervised

bash to_run_cv.sh --dataset esc10
```

## Manual execution training for ESC-10
The different execution argument can be found in the corresponding hydra config file or
by typing `python <script_to_run.py> --help`

```bash
conda activate dct

cd Deep-Co-training/standalone/supervised

python supervised.py \
    -cn ../../config/supervised/supervised_esc.yaml \
    dataset.dataset=esc10 \
    model.model=wideres28_2 
```
<!--
# Reproduction
The directory `standalone/` contains the different script to execute the semi-supervised method and the usual supervised approach. Each approach has its working directory. Each of them contains a *python script*, the *to_run_gs.sh*, and the *to_run.sh*

### Example on how to execute the supervised variant for the UrbanSound8k dataset
```bash
conda activate dct

cd standalone/supervised
bash supervised.sh --dataset ubs8k \
                   --model wideresnet28_2 \
                   --supervised_ratio 1.0 \
                   --epoch 100 \
                   --learning_rate 0.003 \
                   --batch_size 64 \
                   --num_classes 10
                   -C \ # Perform the complete cross_validation
                   -R \ # Resume the training using the last epoch saved
```

Each methods and each dataset have specific parameters, the configuration used to reproduce the result from the paper are describe yaml configuration files that can be found here: `DCT/util/config/`

You can perform training using one of this configuration file. You still need to specify the dataset and the model to use. To do so, you need to directly use the python script and use the *--config* parameter. it will then overide parameters that is define in the config file.

```bash
conda activate dct

cd standalone/supervised
python supervised.py -d ubs8k -m wideresnet28_2 \
                     --config ../DCT/util/config/ubs8k/100_supervised.yml
```
-->
