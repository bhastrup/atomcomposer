#!/bin/bash
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=a100 # sm3090el8
#SBATCH -N 1-1      # Minimum of 1 node
#SBATCH -n 8        # 8 MPI processes per node
#SBATCH --ntasks-per-node=8     # 8 MPI processes per node
#SBATCH --time=0-05:00:00       # Runtime limit
#SBATCH --gres=gpu:1 # RTX3090:1    # Request 1 GPU
#SBATCH --output=slurm_outputs/analyse/exp1/feedback_ent15/%a_%A.out  # % A=job ID, %a=array index
#SBATCH --error=slurm_outputs/analyse/exp1/feedback_ent15/%a_%A.err   # % A=job ID, %a=array index
#SBATCH --job-name=%a  # Use SLURM array ID instead
#SBATCH --array=0-2  # 15 total jobs (3 seeds × 5 run names)

# Calculate indices for both run names and seeds
RUN_NAMES=("final-ent15-AV") # "final-ent15-A" "final-ent15-F" "final-ent15-FV" "final-ent15-AFV")
SEEDS=("seed_0" "seed_1" "seed_2")

#RUN_INDEX=$((SLURM_ARRAY_TASK_ID % 5))  # Maps 0-14 to 0-4 for RUN_NAMES index
#SEED=$((SLURM_ARRAY_TASK_ID / 5))       # Maps 0-14 to 0-2 for seeds

RUN_INDEX=$((SLURM_ARRAY_TASK_ID % ${#RUN_NAMES[@]}))  # Dynamically adjust modulo
SEED=$((SLURM_ARRAY_TASK_ID / ${#RUN_NAMES[@]}))       # Adjust for seeds

scontrol update job=$SLURM_JOB_ID name=feedback-${RUN_NAMES[$RUN_INDEX]}

# Construct RUN_DIR using both indices
RUN_DIR="runs/${RUN_NAMES[$RUN_INDEX]}/${SEEDS[$SEED]}"

# Update MODEL_NAME to use SEED_INDEX instead of SLURM_ARRAY_TASK_ID
# MODEL_NAME="pretrain_run-${SEED}_CP-6_steps-15000.model"
MODEL_NAME="pretrain_run-${SEED}_CP-11_steps-27500.model"
TAG="EXP1_27500"
NUM_EPISODES_CONST=10000
LOG_NAME="pretrain_run-${SEED}.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp1_SB/sample.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME
