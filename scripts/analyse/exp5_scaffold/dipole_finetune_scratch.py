import os
from datetime import datetime
import numpy as np
import submitit


from scripts.train.train import train as submit_fn
from src.tools.launch_utils import (
    load_default_configuration,
    get_sublog_name,
    generate_parameter_combinations,
    submit_jobs,
)


def main() -> None:

    wandb_group = (
        "A-REPO-CLEAN"  # Specify the name of this training, e.g. indicate which agent is used: A, AV, AFV, FV, F etc.
    )

    # Section A: Config and optional rerun
    # use_old_config: load config from the run (e.g. for finetuning with same base). Does not submit by itself.
    # rerun_only: submit a single job immediately and exit (e.g. to extend a crashed run). Set load_model and use_old_config.
    load_model = "runs/nat-com-training/A/seed_0/models/pretrain_run-0_CP-12_steps-30000.model"
    # 'runs/Agent-A-30k-Schedule2500/seed_0/models/pretrain_run-0_steps-30250.model' #
    load_latest = False  # False
    use_old_config = True
    rerun_only = False  # True only for pure reruns (e.g. training crashed)

    config = load_default_configuration(use_old_config, load_model)  # Load default or config from run.
    if rerun_only and (load_model or load_latest):
        config["load_model"] = load_model
        config["load_latest"] = load_latest
        config["safe_xtb"] = False  # Can be set to True, if XTB for some reason caused everything to crash.
        submit_jobs(
            submit_fn=submit_fn, executor=None, parameter_dicts=[config], ask_permission=False, use_submitit=False
        )
        return

    # Section B: Compute resources
    experiment_name = wandb_group
    dt_string = datetime.now().strftime("%d_%m_%H_%M")
    executor = submitit.AutoExecutor(folder=get_sublog_name(experiment_name, dt_string))
    if executor._executor.__class__.__name__ == "SlurmExecutor":
        executor.update_parameters(
            slurm_partition="my_gpu_partition_name",  # Select the GPU partition on your university SLURM cluster.
            gpus_per_node=1,
            tasks_per_node=1,
            cpus_per_task=8,
            slurm_nodes=1,
            slurm_time="7-00:00:00",  # 7 days used here
        )
    elif executor._executor.__class__.__name__ == "LocalExecutor":
        # For basic testing/developing on your 'LocalExecuter' (i.e. your laptop):
        executor.update_parameters(
            timeout_min=40,  # units of minutes
            gpus_per_node=1,
            cpus_per_task=14,
            nodes=1,
            tasks_per_node=1,
            mem_gb=12,
        )
        jobs_args = dict(
            # Specify simpler job args here, e.g. smaller batch sizes, network widths, save_to_wandb=False, etc.
            # But make sure not to overwrite later.
        )
        config.update(jobs_args)

    # Section C: Sweep parameter names and values
    sweep_params = dict(
        seed=[0]  # , 1, 2]
    )

    # Section D: Other parameters to be included in all runs
    config.update(
        dict(
            # Wandb
            save_to_wandb=False,
            entity=os.environ.get(
                "WANDB_ENTITY", ""
            ),  # Set via: export WANDB_ENTITY=your_username  (or set save_to_wandb=False)
            wandb_mode="online",
            wandb_project="isomer-discovery",
            train_mode="dipole-finetuning",
            wandb_job_type="",
            name=experiment_name,
            wandb_group=wandb_group,
            # Data
            mol_dataset="QM7",
            energy_unit="eV",
            # Environment
            min_reward=-3,
            safe_xtb=False,
            # Specify reward coefficients
            # reward_coefs = {'rew_atomisation': 1.0}, 						                        # A
            # reward_coefs = {'rew_formation': 1.0} 							                        # F
            # reward_coefs = {'rew_atomisation': 1.0, 'rew_valid': 3.0} 					            # AV
            # reward_coefs = {'rew_formation': 1.0, 'rew_valid': 3.0}					            # FV
            # reward_coefs = {'rew_atomisation': 1.0, 'rew_formation': 1.0, 'rew_valid': 3.0} 		# AFV
            # reward_coefs = {'rew_atomisation': 1.0, 'rew_valid': 3.0, 'rew_dipole': 1.0}, 		# AVD
            reward_coefs={"rew_atomisation": 1.0, "rew_dipole": 0.0},  # AD
            # Agent policy
            # model = 'painn',
            # rms_norm_update = False,
            # num_interactions=3,
            # network_width=128,
            # Model IO
            device="cuda",
            checkpoints=[int(c) for c in np.arange(0, 1000) * int(100)],
            load_model=load_model,
            save_freq=25,
            keep_models=True,
            # Training scheme
            # rl_algo_online='PPO',
            # learning_rate=2e-5,
            # mini_batch_size=256,
            # num_epochs=1000,
            # Evaluation
            eval_freq=15000,
            eval_freq_fast=100,
        )
    )

    # Section E: Online RL parameters
    config["config_ft"].update(
        dict(
            max_num_steps=int(3e7),
            device=config["device"],
            num_steps_per_iter=512,
            entropy_coef=0.15,
            # max_num_train_iters=1,
            entropy_schedule=dict(
                start_value=0.15,
                final_value=0.25,
                start_iter=0,
                end_iter=30000,
            ),
            reward_coef_schedule=dict(
                schedules={"rew_dipole": (0.0, 0.5)},
                start_iter=30000,  # matches start_num_iter of loaded checkpoint
                end_iter=31000,  # ramp over 2500 iterations
            ),
        )
    )

    # Section F: Create parameter combinations
    parameter_dicts = generate_parameter_combinations(config, sweep_params=sweep_params, exp_name=experiment_name)

    # Section G: Submit jobs
    submit_jobs(
        submit_fn=submit_fn,
        executor=executor,
        parameter_dicts=parameter_dicts,
        ask_permission=False,  # If True, it will ask you to verify the submission before it either submits it via SLURM or executes it directly.
        use_submitit=False,
    )


if __name__ == "__main__":
    main()
