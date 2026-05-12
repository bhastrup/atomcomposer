
# # Set default values for argparse options
# RUN_DIR="from_niflheim/Atom/name_Atom"
# MODEL_NAME="pretrain_run-0_CP-4_steps-10000.model"
# TAG="exp2_10K_prop1"
# LOG_NAME="pretrain_run-0.json"
# PROP_FACTOR=1

# Set default values for argparse options
RUN_DIR="from_niflheim/feedback/ENT-15/final-ent15-AV-with-exp2/seed_0"
MODEL_NAME="pretrain_run-0_CP-6_steps-15000.model"
TAG="EVAL_p100"
PROP_FACTOR=100
LOG_NAME="pretrain_run-0.json"


# Set the flags for the plots to be made (booleans)
MAKE_DISCOVERY_PLOTS=False

MAKE_ENERGY_PLOTS=False
RAE_NAME="rae"

MAKE_ETKDG_PLOTS=True
N_CONFS=1 # Number of conformers to sample for ETKDG
F_MAX=0.5 # Force threshold for relaxations (RL and ETKDG)
STEP_MAX=3 # Max relaxations iterations (RL and ETKDG)



# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp2_MB/make_plots.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --log_name $LOG_NAME \
    --prop_factor $PROP_FACTOR \
    --make_discovery_plots $MAKE_DISCOVERY_PLOTS \
    --make_energy_plots $MAKE_ENERGY_PLOTS \
    --rae_name $RAE_NAME \
    --make_etkdg_plots $MAKE_ETKDG_PLOTS \
    --n_confs $N_CONFS \
    --f_max $F_MAX \
    --step_max $STEP_MAX
