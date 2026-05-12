#!/bin/bash
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=a100 #sm3090_devel #sm3090el8
#SBATCH --gres=gpu:1 # Request 1 GPU (can increase for more) # RTX3090
#SBATCH -N 1-1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --ntasks-per-node=8     # 8 MPI processes per node
#SBATCH --time=0-02:00:00 # 2 days of runtime (can be set to 7 days)
#SBATCH --output=slurm_outputs/analyse/exp3/mio15/%x_%j.out
#SBATCH --error=slurm_outputs/analyse/exp3/mio15/%x_%j.err



STEP_MAX=15000000


# Discovery by formula plot
MAKE_DISCOVERY_BY_FORMULA_PLOT=True

# Scatter plot
MAKE_SCATTER_PLOT=True

# Time series plot
MAKE_TIME_SERIES_PLOT=True
AGGREGATE_ACROSS_FORMULAS=True


# Reward names
REW_NAMES=("AV") #  "A" "F" "FV" "AFV")  # Removed commas between array elements

for REW_NAME in ${REW_NAMES[@]}; do
    for SEED in 0 1 2; do
        BASE_DIR="runs/final-ent15-${REW_NAME}"
        RUN_DIR="${BASE_DIR}/seed_${SEED}"
        echo "Processing seed ${SEED}..."

        python scripts/analyse/exp3_CUM/plot.py \
            --run_dir $RUN_DIR \
            --step_max $STEP_MAX \
            --make_discovery_by_formula_plot $MAKE_DISCOVERY_BY_FORMULA_PLOT \
            --make_scatter_plot $MAKE_SCATTER_PLOT \
            --make_time_series_plot $MAKE_TIME_SERIES_PLOT \
            --aggregate_across_formulas $AGGREGATE_ACROSS_FORMULAS
    done
done
