#!/bin/bash

# ___________________________________________________________________________________ #
die() {
    printf '%s\n' "$1" >& 2
    exit 1
}

parse_long() {
    if [ "$1" ]; then
        echo $1
    else
        die "missing argument value"
    fi
}

function show_help {
    echo "usage:  $BASH_SOURCE [--dataset] [-model] [--ratio] [--epoch] \
        [--learning_rate] [--lambda_cot_max] [--lambda_diff_max] [--lambda_sup_max] \
        [-R RESUME] [-n NODE]"
    echo "    --dataset DATASET                (default ubs8k)"
    echo "    --model MODEL                    (default cnn03)"
    echo "    --ratio RATIO                    (default 0.1)"
    echo "    --epoch EPOCH                    (default 200)"
    echo "    --learning_rate LEARNING_RATE    (default 0.003)"
    echo "    --lambda_cot_max L_COT_MAX       (default 10)"
    echo "    --lambda_diff_max L_DIFF_MAX     (default 0.5)"
    echo "    --lambda_sup_max L_SUP_MAX       (default 1.0)"
    echo ""
    echo "    -R RESUME (default FALSE)"
    echo "    -h help"
    echo "    -? help"

    echo "Osirm parameters"
    echo "    -n NODE (default gpu-nc07)"
    echo ""
    echo "Available datasets"
    echo "    ubs8k"
    echo "    cifar10"
    echo ""
    echo "Available models"
    echo "    cnn0"
    echo "    cnn03"
    echo "    scallable1"
}

# default parameters
DATASET=ubs8k
MODEL=cnn03
RATIO=0.1
EPOCH=200
LEARNING_RATE=0.003
L_COT_MAX=10
L_DIFF_MAX=0.5
L_SUP_MAX=1
RESUME=0
NODE=" "

while :; do
    # If no more option (o no option at all)
    if ! [ "$1" ]; then break; fi

    case $1 in
        -h | -\? | --help) show_help; exit 1;;
        -n | --node) NODE=$(parse_long $2); shift; shift;;
        --dataset) DATASET=$(parse_long $2); shift; shift;;
        --model) MODEL=$(parse_long $2); shift; shift;;
        --ratio) RATIO=$(parse_long $2); shift; shift;;
        --epoch) EPOCH=$(parse_long $2); shift; shift;;
        --learning_rate) LEARNING_RATE=$(parse_long $2); shift; shift;;
        --lambda_cot_max) L_COT_MAX=$(parse_long $2); shift; shift;;
        --lambda_diff_max) L_DIFF_MAX=$(parse_long $2); shift; shift;;
        --lambda_sup_max) L_SUP_MAX=$(parse_long $2); shift; shift;;
        -R) RESUME=1; shift;;
        -?*) echo "WARN: unknown option" $1 >&2
    esac
done

if [ "${NODE}" = " " ]; then
   NODELINE=""
else
    NODELINE="#SBATCH --nodelist=${NODE}"
fi

folds="-t 1 2 3 4 5 6 7 8 9 -v 10"

# ___________________________________________________________________________________ #
LOG_DIR="logs"
SBATCH_JOB_NAME=dct_${DATASET}_${MODEL}

cat << EOT > .sbatch_tmp.sh
#!/bin/bash
#SBATCH --job-name=${SBATCH_JOB_NAME}
#SBATCH --output=${LOG_DIR}/${SBATCH_JOB_NAME}.out
#SBATCH --error=${LOG_DIR}/${SBATCH_JOB_NAME}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
$NODELINE


# sbatch configuration
container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dct/bin/python
script=../co-training/co-training.py

tensorboard_path_root="--tensorboard_path ../../tensorboard/${DATASET}/deep-co-training/${MODEL}/${RATIO}S"
checkpoint_path_root="--checkpoint_path ../../model_save/${DATASET}/deep-co-training"

# ___________________________________________________________________________________ #
parameters=""

# -------- tensorboard and checkpoint path --------
tensorboard_path="\${tensorboard_path_root}/${MODEL}/${RATIO}S"
checkpoint_path="\${checkpoint_path_root}/${MODEL}/${RATIO}S"
parameters="\${parameters} \${tensorboard_path} \${checkpoint_path}"

# -------- dataset --------
parameters="\${parameters} --dataset ${DATASET}"

# -------- model --------
parameters="\${parameters} --model ${MODEL}"

# -------- training parameters --------
parameters="\${parameters} --supervised_ratio ${RATIO}"
parameters="\${parameters} --nb_epoch ${EPOCH}"
parameters="\${parameters} --learning_rate ${LEARNING_RATE}"

# -------- augmentations --------

# -------- resume training --------
if [ $RESUME -eq 1 ]; then
    echo "$RESUME"
    parameters="\${parameters} --resume"
fi

echo python co-training.py ${folds} \${parameters}
srun -n 1 -N 1 singularity exec \${container} \${python} \${script} ${folds} \${parameters}

EOT

echo "sbatch store in .sbatch_tmp.sh"
sbatch .sbatch_tmp.sh

