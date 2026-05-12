RUN_DIR="runs/nat-com-training/A/seed_0"
MODEL_NAME="pretrain_run-0_CP-12_steps-30000.model"
LOG_NAME="pretrain_run-0.json"
TAG="exp6_sample_time"
N_MOLECULES=5

python scripts/analyse/exp6_sample_time/sample_time.py \
    --run_dir $RUN_DIR \
    --model_name $MODEL_NAME \
    --log_name $LOG_NAME \
    --tag $TAG \
    --n_molecules $N_MOLECULES
