#!/bin/bash
#SBATCH --job-name=dct_ubs8k_wideresnet28_2_0.0005lr_10lcm_0.5ldm_160wl
#SBATCH --output=logs/dct_ubs8k_wideresnet28_2_0.0005lr_10lcm_0.5ldm_160wl.out
#SBATCH --error=logs/dct_ubs8k_wideresnet28_2_0.0005lr_10lcm_0.5ldm_160wl.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding



# sbatch configuration
# container=/logiciels/containerCollections/CUDA10/pytorch.sif
container=/users/samova/lcances/container/pytorch-dev.sif
python=/users/samova/lcances/.miniconda3/envs/pytorch-dev/bin/python
script=../co-training/co-training.py

commun_args=""
commun_args="${commun_args} --seed 1234"
commun_args="${commun_args} --model wideresnet28_2"
commun_args="${commun_args} --supervised_ratio 0.1 --learning_rate 0.0005"
commun_args="${commun_args} --batch_size 100 --nb_epoch 300"
commun_args="${commun_args} --lambda_cot_max 10 --lambda_diff_max 0.5 --warmup_length 160"
commun_args="${commun_args} --tensorboard_path deep-co-training_grid-search"
commun_args="${commun_args} --tensorboard_sufix 0.0005lr_160wl_10lcm_0.5ldm"

echo "commun args"
echo 

# Run ubs8K models
run_ubs8k() {
    dataset_args="--dataset ubs8k --num_classes 10 -t 1 2 3 4 5 6 7 8 9 -v 10"

    srun -n 1 -N 1 singularity exec ${container} ${python} ${script} ${commun_args} ${dataset_args}
}

# Run esc10 models
run_esc10() {
    dataset_args="--dataset esc10 --num_classes 10 -t 1 2 3 4 -v 5"

    srun -n 1 -N 1 singularity exec ${container} ${python} ${script} ${commun_args} ${dataset_args}
}

# Run esc50 models
run_esc50() {
    dataset_args="--dataset esc50 --num_classes 50 -t 1 2 3 4 -v 5"

    srun -n 1 -N 1 singularity exec ${container} ${python} ${script} ${commun_args} ${dataset_args}
}

# Run speechcommads models
run_speechcommand() {
    dataset_args="--dataset SpeechCommand --num_classes 35"

    srun -n 1 -N 1 singularity exec ${container} ${python} ${script} ${commun_args} ${dataset_args}
}

case ubs8k in
    ubs8k) run_ubs8k; exit 0;;
    esc10) run_esc10; exit 0;;
    esc50) run_esc50; exit 0;;
    speechcommand | SpeechCommand) run_speechcommand; exit 0;;
    ?*) die "this dataset is not available"; exit 1;;
esac

