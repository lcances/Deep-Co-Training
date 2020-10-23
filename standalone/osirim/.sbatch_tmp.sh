#!/bin/bash
#SBATCH --job-name=sup_SpeechCommand_wideresnet28_2_1.00S
#SBATCH --output=logs/sup_SpeechCommand_wideresnet28_2_1.00S.out
#SBATCH --error=logs/sup_SpeechCommand_wideresnet28_2_1.00S.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding



# sbatch configuration
# container=/logiciels/containerCollections/CUDA10/pytorch.sif
container=/users/samova/lcances/container/pytorch-dev.sif
python=/users/samova/lcances/.miniconda3/envs/pytorch-dev/bin/python
script=../full_supervised/full_supervised.py

# prepare cross validation parameters
# ---- default, no crossvalidation
if [ "SpeechCommand" = "ubs8k" ]; then
    folds=("-t 1 2 3 4 5 6 7 8 9 -v 10")
elif [ "SpeechCommand" = "esc10" ] || [ "SpeechCommand" = "esc50" ]; then
    folds=("-t 1 2 3 4 -v 5")
elif [ "SpeechCommand" = "SpeechCommand" ]; then
    folds=("-t 1 -v 2") # fake array to just ensure at least one run. Not used by the dataset anyway
fi

# if crossvalidation is activated
if [ 0 -eq 1 ]; then
    if [ "SpeechCommand" = "ubs8k" ]; then
        mvar=$(python -c "import DCT.util.utils as u; u.create_bash_crossvalidation(10)")
        IFS=";" read -a folds <<< $mvar

    elif [ "SpeechCommand" = "esc10" ] || [ "SpeechCommand" = "esc50" ]; then
        mvar=$(python -c "import DCT.util.utils as u; u.create_bash_crossvalidation(5)")
        IFS=";" read -a folds <<< $mvar
    fi
fi

# -------- dataset & model ------
common_args="${common_args} --dataset SpeechCommand"
common_args="${common_args} --model wideresnet28_2"

# -------- training common_args --------
common_args="${common_args} --supervised_ratio 1.00"
common_args="${common_args} --nb_epoch 100"
common_args="${common_args} --learning_rate 0.003"
common_args="${common_args} --batch_size 64"
common_args="${common_args} --seed 1234"

# -------- resume training --------
if [ 0 -eq 1 ]; then
    echo "0"
    common_args="${common_args} --resume"
fi

# -------- dataset specific parameters --------
case SpeechCommand in
    ubs8k | esc10) dataset_args="--num_classes 10";;
    esc50) dataset_args="--num_classes 50";;
    SpeechCommand) dataset_args="--num_classes 35";;
    ?*) echo "dataset SpeechCommand is not available"; exit 1;;
esac


run_number=0
for i in ${!folds[*]}
do
    run_number=$(( $run_number + 1 ))

    if [ 0 -eq 1 ]; then
        tensorboard_sufix="--tensorboard_sufix run${run_number}"
    else
        tensorboard_sufix=""
    fi
    
    extra_params="${tensorboard_sufix} ${folds[$i]}"
    
    echo srun -n 1 -N 1 singularity exec ${container} ${python} ${script} ${common_args} ${dataset_args} ${extra_params}
    srun -n 1 -N 1 singularity exec ${container} ${python} ${script} ${common_args} ${dataset_args} ${extra_params}
done

