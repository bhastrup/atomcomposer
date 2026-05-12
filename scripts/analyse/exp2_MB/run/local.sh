
# Set default values for argparse options
RUN_DIR="from_niflheim/feedback/ENT-15/final-ent15-AV-with-exp2/seed_0" #"runs/Atom/name_Atom"
MODEL_NAME="pretrain_run-0_CP-6_steps-15000.model"
TAG="EVAL_p100"
PROP_FACTOR=100
LOG_NAME="pretrain_run-0.json"
RELAX=False

# Pass the arguments from the bash script to the Python script
python scripts/analyse/exp2_MB/sample.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --tag $TAG \
    --prop_factor $PROP_FACTOR \
    --log_name $LOG_NAME \
    --relax $RELAX
