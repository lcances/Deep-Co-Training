# Comparison of Deep Co-Training and Mean Teacher for Audio tagging

Official implementation of the paper 

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