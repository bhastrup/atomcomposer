
# # Set default values for argparse options
# RUN_DIR="runs/A-30k-cap-0.5/seed_0"
# MODEL_NAME="pretrain_run-0_steps-31000.model"
# TAG="thursday-new-31000"
# NUM_EPISODES_CONST=500
# LOG_NAME="pretrain_run-0.json"

# # Pass the arguments from the bash script to the Python script
# python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
#     --run_dir $RUN_DIR \
#     --model_name $MODEL_NAME \
#     --tag $TAG \
#     --num_episodes_const $NUM_EPISODES_CONST \
#     --log_name $LOG_NAME


# # Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms_relaxed.traj})
# python scripts/analyse/exp5_scaffold/visuals/visualize.py \
#     --run_dirs $RUN_DIR \
#     --tag $TAG \
#     --sorting_key dipole_relaxed \
#     --smiles_col NEW_SMILES \
#     --n_mols 3



# ################################################################################

# # Set default values for argparse options
# RUN_DIR="runs/A-30k-cap-0.5/seed_0"
# MODEL_NAME="pretrain_run-0_steps-32000.model"
# TAG="thursday-new-32000"
# NUM_EPISODES_CONST=500
# LOG_NAME="pretrain_run-0.json"

# # Pass the arguments from the bash script to the Python script
# python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
#     --run_dir $RUN_DIR \
#     --model_name $MODEL_NAME \
#     --tag $TAG \
#     --num_episodes_const $NUM_EPISODES_CONST \
#     --log_name $LOG_NAME


# # Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms_relaxed.traj})
# python scripts/analyse/exp5_scaffold/visuals/visualize.py \
#     --run_dirs $RUN_DIR \
#     --tag $TAG \
#     --sorting_key dipole_relaxed \
#     --smiles_col NEW_SMILES \
#     --n_mols 3


################################################################################

# Set default values for argparse options
RUN_DIR="runs/A-30k-cap-0.5/seed_0"
MODEL_NAME="pretrain_run-0_steps-33000.model"
TAG="thursday-new-33000"
NUM_EPISODES_CONST=500
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp5_scaffold/dipole_evaluate.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME


# Visualize results (expects: <RUN_DIR>/results/<TAG>/<formula>/{df.csv,atoms.traj})
python scripts/analyse/exp5_scaffold/visuals/visualize.py \
    --run_dirs $RUN_DIR \
    --tag $TAG \
    --sorting_key dipole_relaxed \
    --smiles_col NEW_SMILES \
    --n_mols 3

