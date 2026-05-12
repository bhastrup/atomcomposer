"""
Plot learning curves from training logs saved to disk.

Reads the JSON-lines files written by InfoSaver during training
(runs/<agent>/seed_<n>/results/train_run-<n>_train.txt) and plots
key metrics vs environment steps.  No W&B account needed.

Usage
-----
  # Single agent, all seeds found automatically:
  python scripts/analyse/plot_learning_curves.py runs/Agent-A

  # Multiple agents on the same figure:
  python scripts/analyse/plot_learning_curves.py runs/Agent-A runs/Agent-F runs/Agent-AV runs/Agent-FV runs/Agent-AFV

  # Custom smoothing window and output path:
  python scripts/analyse/plot_learning_curves.py runs/Agent-A --smooth 200 --out figs/curves.png

  # Also export smoothed data as CSV:
  python scripts/analyse/plot_learning_curves.py runs/Agent-A --csv curves.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


METRICS = {
    "final:Valid": "Validity rate",
    "return_mean": "Mean return",
    "final:AE": "Atomization energy (eV)",
    "final:RAE": "Relative AE (eV)",
}

AGENT_COLORS = {
    "Agent-A": "#1f77b4",
    "Agent-F": "#ff7f0e",
    "Agent-AV": "#2ca02c",
    "Agent-FV": "#d62728",
    "Agent-AFV": "#9467bd",
}


def smooth_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average."""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def smooth_rolling(values: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(values)
    return s.rolling(window, min_periods=1, center=True).mean().to_numpy()


def read_log(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "total_num_steps" not in df.columns:
        return pd.DataFrame()
    df = df.sort_values("total_num_steps").reset_index(drop=True)
    return df


def find_train_logs(run_dir: Path) -> List[Path]:
    """Find all *_train.txt files under run_dir/seed_*/results/."""
    return sorted(run_dir.glob("seed_*/results/*_train.txt"))


def load_agent(run_dir: Path) -> Optional[pd.DataFrame]:
    logs = find_train_logs(run_dir)
    if not logs:
        print(f"  No training logs found under {run_dir}", file=sys.stderr)
        return None
    frames = []
    for log in logs:
        df = read_log(log)
        if not df.empty:
            seed = log.parts[-3]  # seed_0, seed_1, ...
            df["seed"] = seed
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def plot(
    agent_data: Dict[str, pd.DataFrame],
    metrics: List[str],
    smooth: int,
    out: Path,
    csv_out: Optional[Path],
) -> None:
    present = [m for m in metrics if any(m in df.columns for df in agent_data.values())]
    if not present:
        print("None of the requested metrics were found in the logs.", file=sys.stderr)
        sys.exit(1)

    n_metrics = len(present)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4), squeeze=False)
    axes = axes[0]

    csv_rows = []

    for agent_name, df in agent_data.items():
        color = AGENT_COLORS.get(agent_name, None)
        seeds = df["seed"].unique()

        for ax, metric in zip(axes, present):
            if metric not in df.columns:
                continue

            seed_curves = []
            for seed in seeds:
                sdf = df[df["seed"] == seed].dropna(subset=[metric])
                if sdf.empty:
                    continue
                steps = sdf["total_num_steps"].to_numpy()
                vals = sdf[metric].to_numpy()
                smoothed = smooth_rolling(vals, smooth)
                seed_curves.append((steps, smoothed))
                csv_rows.append(
                    pd.DataFrame(
                        {"agent": agent_name, "seed": seed, "metric": metric, "steps": steps, "value": smoothed}
                    )
                )

            if not seed_curves:
                continue

            # Plot individual seeds as thin lines, mean as thick line
            for steps, smoothed in seed_curves:
                ax.plot(steps, smoothed, alpha=0.3, linewidth=0.8, color=color)

            if len(seed_curves) > 1:
                # Interpolate onto the shared step range (intersection across seeds)
                common = np.linspace(
                    max(s[0] for s, _ in seed_curves),
                    min(s[-1] for s, _ in seed_curves),
                    num=500,
                )
                interped = np.stack([np.interp(common, s, v) for s, v in seed_curves])
                ax.plot(common, interped.mean(axis=0), label=agent_name, linewidth=2, color=color)
            else:
                ax.plot(seed_curves[0][0], seed_curves[0][1], label=agent_name, linewidth=2, color=color)

    for ax, metric in zip(axes, present):
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(METRICS.get(metric, metric))
        ax.set_title(METRICS.get(metric, metric))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out}")

    if csv_out and csv_rows:
        pd.concat(csv_rows, ignore_index=True).to_csv(csv_out, index=False)
        print(f"Saved CSV to {csv_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training learning curves from disk logs.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directories (e.g. runs/Agent-A runs/Agent-F)")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(METRICS.keys()),
        help="Metrics to plot (default: all known metrics)",
    )
    parser.add_argument("--smooth", type=int, default=100, help="Rolling-mean window width (default: 100)")
    parser.add_argument("--out", type=Path, default=Path("learning_curves.png"), help="Output figure path")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV export path for smoothed data")
    args = parser.parse_args()

    agent_data = {}
    for run_dir in args.run_dirs:
        if not run_dir.exists():
            print(f"Warning: {run_dir} does not exist, skipping.", file=sys.stderr)
            continue
        agent_name = run_dir.name
        print(f"Loading {agent_name} ...")
        df = load_agent(run_dir)
        if df is not None:
            agent_data[agent_name] = df
            print(f"  {len(df)} rows, seeds: {sorted(df['seed'].unique())}")

    if not agent_data:
        print("No data found. Exiting.", file=sys.stderr)
        sys.exit(1)

    plot(agent_data, args.metrics, args.smooth, args.out, args.csv)


if __name__ == "__main__":
    main()
