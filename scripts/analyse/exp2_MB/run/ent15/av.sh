#!/bin/bash
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=a100 # sm3090el8
#SBATCH -N 1         # 1 node per job
#SBATCH -n 8         # 8 MPI processes per job
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:1 # RTX3090
#SBATCH --output=slurm_outputs/analyse/exp2/feedback_ent15/AV/seed_%A_%a.out
#SBATCH --error=slurm_outputs/analyse/exp2/feedback_ent15/AV/seed_%A_%a.err
#SBATCH --array=0-2  # Array indices for three jobs (seed_0, seed_1, seed_2)

REW_NAME="AV"

# Define the list of seeds
SEEDS=("seed_0" "seed_1" "seed_2")

# Set default values for argparse options
TAG="EXP2_p100_RELAX"
PROP_FACTOR=100

CP_NUM=11
GRAD_STEPS=27500

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}" # Get the seed for this job
LOG_NAME="pretrain_run-${SLURM_ARRAY_TASK_ID}.json"
RUN_DIR="runs/final-ent15-${REW_NAME}/$SEED"
MODEL_NAME="pretrain_run-${SLURM_ARRAY_TASK_ID}_CP-${CP_NUM}_steps-${GRAD_STEPS}.model"

# Perform optimization
RELAX=True

# Additional flags for controlling behavior
ROLLOUTS_FROM_FILE=True
FEATURES_FROM_FILE=True


# Run the Python script for this seed
python scripts/analyse/exp2_MB/sample.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --prop_factor $PROP_FACTOR \
    --log_name $LOG_NAME \
    --relax $RELAX \
    --rollouts_from_file $ROLLOUTS_FROM_FILE \
    --features_from_file $FEATURES_FROM_FILE \
