import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def default_config() -> dict:
    """
    Returns default training configuration.

    Experiment scripts (e.g. scripts/train/experiments/nat-com-version/a.py) override
    the values they care about via config.update({...}).  These defaults only need to
    cover parameters that are NOT overridden there.

    config_ft holds parameters for the inner PPO loop (training_loop.py).
    The split is a historical artefact and will be flattened in a future refactor.
    """

    config_ft = dict(
        # PPO loop budget
        max_num_steps=int(3e7),
        num_steps_per_iter=512,
        # PPO hyperparameters
        clip_ratio=0.2,
        vf_coef=0.5,
        entropy_coef=0.15,
        target_kl=0.05,
        gradient_clip=0.5,
        max_num_train_iters=7,
        lam=0.97,
        # Optimizer
        optimizer="adam",  # 'adam' | 'amsgrad'
        # Device — overwritten from outer config before training starts
        device="cuda",
        # Optional schedules — set to a dict in the experiment script to activate
        entropy_schedule=None,
        reward_coef_schedule=None,
    )

    config = dict(
        # --- Identity ---
        name="train",
        seed=0,
        # --- Output directories (overwritten by set_directories in launch_utils) ---
        log_dir="logs",
        model_dir="models",
        data_dir="data",
        results_dir="results",
        # --- Compute ---
        device="cuda",
        # --- Data ---
        mol_dataset="QM7",  # 'QM7' | 'QM9'
        energy_unit="eV",
        split_method="read_split_from_disk",
        # --- Environment ---
        formulas=None,
        num_envs=8,
        min_atomic_distance=0.6,
        max_solo_distance=2.0,
        min_reward=-3,
        build_mode="fix_heavy",  # 'fix_heavy' | 'fix_TM' | 'none'
        hydrogen_delay=False,
        no_hydrogen_focus=False,
        relax_steps_final=0,
        safe_xtb=True,
        # --- Reward ---
        reward_coefs={"rew_atomisation": 1.0},
        # --- Agent (PaiNN) ---
        model="painn",
        network_width=128,
        num_interactions=3,
        cutoff=4.0,
        min_mean_distance=1.0,
        max_mean_distance=1.6,
        rms_norm_update=False,
        # --- Model IO ---
        load_model=None,
        load_latest=False,
        checkpoints=[0, 100, 1000],
        save_freq=10,
        keep_models=False,
        # --- Outer training loop ---
        num_epochs=1000,
        rl_algo_online="PPO",
        learning_rate=2e-5,
        mini_batch_size=256,
        train_mode="tabula_rasa",
        # --- Evaluation ---
        eval_freq=1500,
        eval_freq_fast=100,
        launch_eval_new_job=False,  # True submits eval as a separate SLURM job
        # --- Logging ---
        log_level="INFO",
        save_rollouts="none",  # 'none' | 'train' | 'eval' | 'all'
        # --- Weights & Biases (all optional — set save_to_wandb=False to disable) ---
        save_to_wandb=False,
        entity="",  # or set via: export WANDB_ENTITY=your_username
        wandb_project="isomer-discovery",
        wandb_group="",
        wandb_job_type="",
        wandb_name="",
        wandb_mode="online",
        wandb_watch_model=False,
        # --- Inner PPO loop config ---
        config_ft=config_ft,
    )

    return config
