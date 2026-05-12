#!/bin/bash
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=sm3090el8
#SBATCH -N 1-1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --ntasks-per-node=8     # 8 MPI processes per node
#SBATCH --time=0-05:00:00 # 2 days of runtime (can be set to 7 days)
#SBATCH --gres=gpu:RTX3090:1 # Request 1 GPU (can increase for more)
#SBATCH --output=slurm_outputs/analyse/feedback_ent15/exp1/av_%j.out  # %j gives SLURM job ID
#SBATCH --error=slurm_outputs/analyse/feedback_ent15/exp1/av_%j.out  # %j gives SLURM job ID


# Set default values for argparse options
RUN_DIR="runs/final-ent15-AV/seed_0"
MODEL_NAME="pretrain_run-0_CP-6_steps-15000.model"
TAG="feedback-100"
NUM_EPISODES_CONST=10
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp1_SB/sample.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME
