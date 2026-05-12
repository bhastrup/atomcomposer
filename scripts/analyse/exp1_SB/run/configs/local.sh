
# Set default values for argparse options
RUN_DIR="from_niflheim/Atom/name_Atom" #"runs/Atom/name_Atom"
MODEL_NAME="pretrain_run-0_CP-4_steps-10000.model"
TAG="exp1-10k-prop10000-thread"
NUM_EPISODES_CONST=100
LOG_NAME="pretrain_run-0.json"

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp1_SB/sample.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --num_episodes_const $NUM_EPISODES_CONST \
    --log_name $LOG_NAME
