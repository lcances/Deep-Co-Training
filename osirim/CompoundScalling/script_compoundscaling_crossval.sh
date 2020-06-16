#!/bin/bash

if [ "$#" -ne 7 ]; then
  echo "wrong number of parameter"
  echo "received $# arguments"
  echo "usage: ./script_compoundscaling_crossval.sh gpu-nc0x alpha beta gamma phi round_up title"
  exit 1
fi

GPU_NODE=$1
ALPHA=$2
BETA=$3
GAMMA=$4
PHI=$5
ROUND_UP=$6
TITLE=$7

SBATCH_JOB_NAME=CS_$TITLE

cat << EOT > .sbatch_tmp.sh
#!/bin/bash
#SBATCH --job-name=$SBATCH_JOB_NAME
#SBATCH --output=${SBATCH_JOB_NAME}.out
#SBATCH --error=${SBATCH_JOB_NAME}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=GPUNodes
#SBATCH --nodelist=$GPU_NODE
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

PR=$2
MODEL=$3

# ---- Compound scales ----
parameters="--alpha ${ALPHA} --beta ${BETA} --gamma ${GAMMA} --phi ${PHI}"

# ---- round up----
parameters="\${parameters} --round_up ${ROUND_UP}"

# ---- title ----
parameters="\${parameters} --title ${TITLE}"

# Sbatch configuration
container=/logiciels/containerCollections/CUDA10/pytorch.sif
python=/users/samova/lcances/.miniconda3/envs/dl/bin/python
script=../../standalone/CompoundScalling/copoundScalling.py

srun -n1 -N1 singularity exec \${container} \${python} \${script} \${parameters}

EOT

echo "sbatch store in .sbatch_tmp.sh"
sbatch .sbatch_tmp.sh

