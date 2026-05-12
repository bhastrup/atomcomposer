#!/bin/bash
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=sm3090el8
#SBATCH -N 1         # 1 node per job
#SBATCH -n 8         # 8 MPI processes per job
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --output=slurm_outputs/analyse/exp2/feedback_ent15/AV/seed_%a_%A.out
#SBATCH --error=slurm_outputs/analyse/exp2/feedback_ent15/AV/seed_%a_%A.err
#SBATCH --array=0-2  # Array indices for two jobs (seed_1 and seed_2)

# Define the list of seeds
SEEDS=("seed_0" "seed_1" "seed_2")

# Set default values for argparse options
TAG="EVAL_p100"
PROP_FACTOR=100

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}" # Get the seed for this job
LOG_NAME="pretrain_run-${SLURM_ARRAY_TASK_ID}.json"
RUN_DIR="runs/final-ent15-AV/$SEED"
MODEL_NAME="pretrain_run-${SLURM_ARRAY_TASK_ID}_CP-6_steps-15000.model"


# Set the flags for the plots to be made (booleans)
MAKE_DISCOVERY_PLOTS=False
MAKE_ENERGY_PLOTS=False
MAKE_ETKDG_PLOTS=True

# Number of conformers to sample for ETKDG
N_CONFS=5


# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp2_MB/make_plots.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --log_name $LOG_NAME \
    --prop_factor $PROP_FACTOR \
    --make_discovery_plots $MAKE_DISCOVERY_PLOTS \
    --make_energy_plots $MAKE_ENERGY_PLOTS \
    --make_etkdg_plots $MAKE_ETKDG_PLOTS \
    --n_confs $N_CONFS
