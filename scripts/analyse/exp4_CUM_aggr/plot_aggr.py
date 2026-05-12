import argparse
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def parse_cmd():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("filenames", nargs="+", help="List of filenames")
    parser.add_argument("--save_dir", type=str, help="Path to save the plot")
    parser.add_argument("--step_max", type=str, default="None", help="Maximum number of steps to consider")

    return parser.parse_args()


def plot_figure(data, args, rediscovery=False):
    # Plot data
    fig, ax = plt.subplots()
    for name, d in data.items():
        ax.plot(d["steps"], d["counts"], label=name)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Counts")
    ax.legend()

    # Save plot
    plot_path = Path.cwd() / f"{args.save_dir}/exp3_cum/{'rediscovery' if rediscovery else 'discovery'}.png"
    os.makedirs(plot_path.parent, exist_ok=True)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    args = parse_cmd()
    step_max = None if args.step_max == "None" else int(args.step_max)

    print(args.filenames)

    # Load data .npz files
    discovery = {}
    rediscovery = {}
    for filename in args.filenames:
        name = filename.split("/results")[0]
        name = name.split("/")[-2:]
        name = "/".join(name)
        discovery[name] = np.load(filename)

        # filename without extension
        filename = filename.split(".npz")[0]
        filename += "_rediscovery.npz"
        rediscovery[name] = np.load(filename)

    # Plot data
    plot_figure(discovery, args)
    plot_figure(rediscovery, args, rediscovery=True)
