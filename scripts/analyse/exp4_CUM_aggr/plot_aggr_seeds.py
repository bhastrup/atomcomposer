import os
from pathlib import Path

from matplotlib import pyplot as plt

from scripts.analyse.exp4_CUM_aggr.ts_plot import load_time_series_data, plot_time_series_figure
from scripts.analyse.exp4_CUM_aggr.bar_plot import load_discovery_metrics_data, plot_cum_discovery_seeds_2_subplots


FONT_TYPE = "serif"
FONT_SIZE = 10
TITLE_FONT_SIZE = FONT_SIZE + 3

if __name__ == "__main__":
    n_seeds = 3
    step_max = 20000000
    mol_dataset = "qm7"
    ref_value = 6465

    base_dir = Path.cwd() / "from_niflheim/digital_discovery/"  # 'runs'
    exp_name = "entropy-schedule"
    sample_tag = "in_sample"

    plot_aggr_time_series = True
    plot_aggr_final_metrics = True

    run_names_map = {
        f"{exp_name}-A": "A",
        f"{exp_name}-AV": "AV",
        f"{exp_name}-F": "F",
        f"{exp_name}-FV": "FV",
        f"{exp_name}-AFV": "AFV",
    }
    run_names = list(run_names_map.keys())

    for key in run_names_map.keys():
        for seed in range(n_seeds):
            run_dir = base_dir / f"{key}/seed_{seed}"
            assert run_dir.exists(), f"Run directory {run_dir} does not exist"

    plot_dir = Path.cwd() / f"results/{exp_name}/aggr_results_seed_new"
    os.makedirs(plot_dir, exist_ok=True)

    # Plot time series
    if plot_aggr_time_series:
        # Load data
        discovery_ts, rediscovery_ts = load_time_series_data(base_dir, run_names_map, n_seeds)

        # Make plots
        fig_discovery, ax_discovery = plot_time_series_figure(discovery_ts, legends_on=True, ref_value=ref_value)
        fig_rediscovery, ax_rediscovery = plot_time_series_figure(
            rediscovery_ts, rediscovery=True, legends_on=False, ref_value=ref_value
        )

        # Save plot
        fig_discovery.savefig(plot_dir / "ts_discovery.png", dpi=300)
        fig_rediscovery.savefig(plot_dir / "ts_rediscovery.png", dpi=300)

    # Plot final discovery metrics
    if plot_aggr_final_metrics:
        # Load data
        dm_all_seeds = load_discovery_metrics_data(base_dir, run_names_map, n_seeds, step_max, mol_dataset, sample_tag)

        # Make plots
        bar_fig_split, (ax1_split, ax2_split) = plot_cum_discovery_seeds_2_subplots(
            dm_all_seeds, mol_dataset=mol_dataset, ratio=True
        )
        # Save plot
        bar_fig_split.savefig(plot_dir / "cum_discovery_bars.png", dpi=300)

    # Load and plot combined plots
    # combine plots in a row (time series and bar plots)

    horizontal = True
    n_plots = 3

    if horizontal:
        fig, axs = plt.subplots(1, n_plots, figsize=(10, 4), width_ratios=[1, 1, 1])
    else:
        fig, axs = plt.subplots(n_plots, 1, figsize=(8, 8), height_ratios=[1, 1, 1])

    fig_discovery = plt.imread(plot_dir / "ts_discovery.png")
    fig_rediscovery = plt.imread(plot_dir / "ts_rediscovery.png")
    fig_bar = plt.imread(plot_dir / "cum_discovery_bars.png")

    for ax, fig in zip(axs, [fig_discovery, fig_rediscovery, fig_bar]):
        ax.imshow(fig)
        ax.axis("off")

    plt.tight_layout(h_pad=-0.5)  # Adjust spacing between subplots
    plt.savefig(plot_dir / f"combined_plots{'_horizontal' if horizontal else ''}.png", bbox_inches="tight", dpi=300)
