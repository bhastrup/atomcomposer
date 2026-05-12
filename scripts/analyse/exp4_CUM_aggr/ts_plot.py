import copy
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


FONT_TYPE = "serif"
FONT_SIZE = 16
TITLE_FONT_SIZE = FONT_SIZE + 1
figsize = (4, 4)


def get_path_to_time_series(exp_dir: Path, run_name: str, seed: int) -> tuple[Path, Path]:
    return (
        Path(f"{exp_dir}/{run_name}/seed_{seed}/results/discovery_plots/cumulative_aggregated.npz"),
        Path(f"{exp_dir}/{run_name}/seed_{seed}/results/discovery_plots/cumulative_aggregated_rediscovery.npz"),
    )


def load_time_series_data(exp_dir: Path, run_names: list[str], n_seeds: int) -> tuple[dict, dict]:
    """Load data .npz files for each run and seed"""
    discovery_ts = {}
    rediscovery_ts = {}
    for run_name, name_tag in run_names.items():
        path_tuples_list = [get_path_to_time_series(exp_dir, run_name, seed) for seed in range(n_seeds)]
        existing_paths = [
            path_tuple for path_tuple in path_tuples_list if path_tuple[0].exists() and path_tuple[1].exists()
        ]

        if not len(existing_paths) == n_seeds:
            print(f"Missing {n_seeds - len(existing_paths)} paths for {run_name}")
            continue

        discovery_ts[name_tag] = {}
        rediscovery_ts[name_tag] = {}
        for seed in range(n_seeds):
            path_tuple = existing_paths[seed]

            discovery_ts[name_tag][seed] = np.load(path_tuple[0])
            rediscovery_ts[name_tag][seed] = np.load(path_tuple[1])

    return discovery_ts, rediscovery_ts


def plot_time_series_figure(data, rediscovery=False, legends_on=True, ref_value: int = None):
    # Plot data
    fig, ax = plt.subplots(figsize=figsize)
    for name, d in data.items():
        df = pd.DataFrame(columns=["steps"])

        for seed, d_seed in d.items():
            steps = d_seed["steps"]
            counts = d_seed["counts"]

            # Append final step to steps and repeat last count
            final_step = 20000000
            steps = np.append(steps, final_step)
            counts = np.append(counts, counts[-1])

            new_df = pd.DataFrame({"steps": steps, f"counts_{seed}": counts})

            df = pd.merge(df, new_df, on="steps", how="outer")

        # Trim, but make sure to keep the last step.
        last_row = df.iloc[-1].copy()
        df = df.ffill().iloc[::100]
        df = pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)

        df["mean"] = np.mean(df.iloc[:, 1:], axis=1)
        df["std"] = np.std(df.iloc[:, 1:], axis=1, ddof=1)

        ax.plot(df["steps"], df["mean"], label=f"{name}", linewidth=2)
        ax.fill_between(df["steps"], df["mean"] - df["std"], df["mean"] + df["std"], alpha=0.6)

    # Plot reference value
    if ref_value is not None:
        ax.axhline(y=ref_value, color="black", linestyle="--", linewidth=1.5, label="QM7 size")

    ax.grid()
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)

    ax.set_xlabel("Steps", fontsize=FONT_SIZE, font=FONT_TYPE)
    ax.set_ylabel("Counts", fontsize=FONT_SIZE, font=FONT_TYPE)

    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.get_xaxis().get_offset_text().set_fontsize(FONT_SIZE)
    ax.get_yaxis().get_offset_text().set_fontsize(FONT_SIZE)
    ax.get_xaxis().get_offset_text().set_fontfamily(FONT_TYPE)
    ax.get_yaxis().get_offset_text().set_fontfamily(FONT_TYPE)

    if legends_on:
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(ha) for ha in handles]
        [ha.set_linewidth(6) for ha in handles]
        ax.legend(
            handles=handles, labels=labels, loc="upper left", ncol=2, prop={"size": FONT_SIZE - 5, "family": FONT_TYPE}
        )

    ax.set_title(
        "Discovery (Novel)" if not rediscovery else "Rediscovery", fontsize=TITLE_FONT_SIZE, fontfamily=FONT_TYPE
    )

    # make outer frame thicker
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    fig.tight_layout()

    return fig, ax
