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


"""
######################### GUIDE ###################################

    REPRODUCE RESULTS FROM PAPER:
    The hyperparameters shown here reproduce the results from the paper for the agent "AFV".
    See a.py for full documentation of the submission workflow.

"""


def main() -> None:

    wandb_group = "Agent-AFV"

    load_model = None
    load_latest = False
    use_old_config = False

    config = load_default_configuration(use_old_config, load_model)
    if use_old_config and (load_model or load_latest):
        config["load_model"] = load_model
        config["load_latest"] = load_latest
        config["safe_xtb"] = False
        submit_jobs(
            submit_fn=submit_fn, executor=None, parameter_dicts=[config], ask_permission=False, use_submitit=False
        )

    experiment_name = wandb_group
    dt_string = datetime.now().strftime("%d_%m_%H_%M")
    executor = submitit.AutoExecutor(folder=get_sublog_name(experiment_name, dt_string))
    if executor._executor.__class__.__name__ == "SlurmExecutor":
        executor.update_parameters(
            slurm_partition="my_gpu_partition_name",
            gpus_per_node=1,
            tasks_per_node=1,
            cpus_per_task=8,
            slurm_nodes=1,
            slurm_time="7-00:00:00",
        )
    elif executor._executor.__class__.__name__ == "LocalExecutor":
        executor.update_parameters(
            timeout_min=40,
            gpus_per_node=1,
            cpus_per_task=14,
            nodes=1,
            tasks_per_node=1,
            mem_gb=12,
        )

    sweep_params = dict(seed=[0])

    config.update(
        dict(
            save_to_wandb=False,
            entity=os.environ.get("WANDB_ENTITY", ""),
            wandb_mode="online",
            wandb_project="isomer-discovery",
            train_mode="tabula_rasa",
            wandb_job_type="",
            name=experiment_name,
            wandb_group=wandb_group,
            mol_dataset="QM7",
            energy_unit="eV",
            min_reward=-3,
            safe_xtb=False,
            reward_coefs={"rew_atomisation": 1.0, "rew_formation": 1.0, "rew_valid": 3.0},  # AFV
            model="painn",
            rms_norm_update=False,
            num_interactions=3,
            network_width=128,
            device="cuda",
            checkpoints=[int(c) for c in np.arange(0, 51) * int(2500)],
            load_model=load_model,
            save_freq=10,
            rl_algo_online="PPO",
            learning_rate=2e-5,
            mini_batch_size=256,
            num_epochs=1000,
            eval_freq=1500,
            eval_freq_fast=100,
        )
    )

    config["config_ft"].update(
        dict(
            max_num_steps=int(3e7),
            device=config["device"],
            entropy_coef=0.15,
            entropy_schedule=dict(
                start_value=0.15,
                final_value=0.25,
                start_iter=0,
                end_iter=30000,
            ),
            num_steps_per_iter=512,
        )
    )

    parameter_dicts = generate_parameter_combinations(config, sweep_params=sweep_params, exp_name=experiment_name)

    submit_jobs(
        submit_fn=submit_fn,
        executor=executor,
        parameter_dicts=parameter_dicts,
        ask_permission=False,
        use_submitit=False,
    )


if __name__ == "__main__":
    main()
