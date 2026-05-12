
STEP_MAX=20000000

# Discovery by formula plot
MAKE_DISCOVERY_BY_FORMULA_PLOT=False

# Scatter plot
MAKE_SCATTER_PLOT=True

# Time series plot
MAKE_TIME_SERIES_PLOT=True
AGGREGATE_ACROSS_FORMULAS=False



BASE_DIR="from_niflheim/digital_discovery"
EXP_NAME="entropy-schedule"
REW_NAMES=("A") #  "AV" "F" "FV" "AFV")
SEEDS=(0) # 1 2)


for REW_NAME in ${REW_NAMES[@]}; do
    for SEED in ${SEEDS[@]}; do
        EXP_DIR="${BASE_DIR}/${EXP_NAME}-${REW_NAME}"
        RUN_DIR="${EXP_DIR}/seed_${SEED}"
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
