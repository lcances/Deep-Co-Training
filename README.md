# COMPARISON OF DEEP CO-TRAINING AND MEAN-TEACHER APPROACHES FOR SEMI-SUPERVISED AUDIO TAGGING
## Official implementation

Recently, a number of semi-supervised learning (SSL) methods, in the framework of deep learning (DL), were shown to achieve state-of-the-art results on image datasets, while using a (very) limited amount of labeled data. To our knowledge, these approaches adapted and applied to audio data are still sparse, in particular for audio tagging (AT). In this work, we adapted the Deep-Co-Training algorithm (DCT) to perform AT, and compared it to another SSL approach called Mean Teacher (MT), that has been used by the winning participants of the DCASE competitions these last two years. Experiments were performed on three standard audio datasets: Environmental Sound classification (ESC-10), UrbanSound8K, and Google Speech Commands. We show that both DCT and MT achieved performance approaching that of a fully supervised training setting, while using a fraction of the labeled data available, and the remaining data as unlabeled data. In some cases, DCT even reached the best accuracy, for instance, 72.6% using half of the labeled data, compared to 74.4% using all the labeled data. DCT also consistently outperformed MT in almost all configurations. For instance, the most significant relative gains brought by DCT reached 12.2% on ESC-10, compared to 7.6% with MT. Our code is available online

# Requirements

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
