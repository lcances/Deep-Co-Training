# Configuration files
The configuration files exist to make the reproduction of the experiment more straight forward.

To reproduce an experiment, the user can simply move into the `standalone` directory and execute the experiment he want to reproduce as follow:

Exemple to reproduce full supervised results for UrbanSoun8k

```bash
bash standalone/full_supervised.sh -c config/ubs8k/supervised.yml
```

Or to reproduce result from Deep Co-Training uniloss on UrbanSound8k

```bash
bash standalone/co-training/co-training-independant_loss.sh -c config/ubs8k/dct_uniloss.yml
```

Each configuration file must fullfil the following requirement
- Have a `dataset_path` entry
- Have a `dataset` entry
- Specify a `supervised_ratio`
- Specify the `model` to use
- The training parameters such as:
    - `batch_size`
    - nb of `epoch`
    - the `optimizer` function to use (see DCT/<dataset>/train_parameters.py)
- Some log information such as:
    - The directory to save the model `checkpoint_path`
    - The diretory to save the tensorboard logs `tensorboard_path`
    - A custom tensorboard name sufix `tensorboard_sufix`

Depending on the framework use, some extra parameters would be needed
- **Deep Co-Training**
    - Hyperparameter `lambda_cot_max`
    - Hyperparameter `lambda_diff_max`
    - Hyperparameter `lambda_sup_max`
    - Adversarial generation parameter `epsilon`

- **Dee Co-Training Uniloss**
    - Parameters from Deep Co-Training
    - Loss scheduler name `loss_scheduler`
    - Loss scheduler number of `steps`
    - Cosine based loss scheduler number of `cycle`
    - Annealed based loss scheduler decay scale `beta`
    - Minimum draw probability for supervised loss `plsup_mini`