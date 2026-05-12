# AtomComposer: Rediscovering Chemical Space from First Principles with Reinforcement Learning


This repository contains the training and analysis code used in the study:

**AtomComposer: Rediscovering Chemical Space from First Principles with Reinforcement Learning**  
*Bjarke Hastrup, François Cornet, Tejs Vegge, and Arghya Bhowmik*  
Under review at *Nature Communications*.  
Preprint (Version 1) available on [Research Square](https://doi.org/10.21203/rs.3.rs-6900238/v1).


<table>
  <tr>
    <td><img src="resources/image_grid.png" height="300"></td>
    <td><img src="resources/atomcomposer.png" height="300"></td>
  </tr>
</table>

> **Interactive web app:** Once the environment is installed, you can explore the trained agents interactively via a Streamlit app (`streamlit run app_store/About.py`). See the [Web app](#web-app) section for details.

---

<details>
<summary><b>Installation Guide</b></summary>

### 1. Install Conda (if not already installed)
Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/bhastrup/isomer-discovery-rl.git
cd isomer-discovery-rl
```

### 3. Install Mamba (if not already installed)
```bash
conda install -n base -c conda-forge mamba
```

### 4. Create the Environment using Mamba
Run the following command to create the environment from `env.yaml`:
```bash
mamba env create -f env.yaml
```

### 5. Activate the Environment
Once the installation is complete, activate the environment:
```bash
eval "$(mamba shell hook --shell bash)"
mamba activate atomcomposer
```

### 6. Verify Installation
Check that the required packages are installed:
```bash
python -c 'import torch, ase, rdkit, streamlit, pandas; print("✅ All core packages are working!")'
```

### 7. Set Up PYTHONPATH (One-Time)

To ensure Python can locate the `src/` package, set the `PYTHONPATH` automatically when activating the environment. Run the following **from the root of the project directory** (where `src/` is located):

```bash
# Set PYTHONPATH when the Conda environment is activated
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export PYTHONPATH=\$(pwd)" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Unset it on deactivate
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo "unset PYTHONPATH" > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```
and reactivate the *atomcomposer*.

</details>

---

<details>
<summary><b>Dataset preparation</b></summary>

For QM7 training, run the following commands (takes 1-2 minutes):
```bash
python scripts/prep/preprocess_data.py
python scripts/prep/split_train_test.py
```
This saves a dataset of QM7 structures into the *data/* folder (along with XTB energies, SMILES representations and additional meta-data).
QM9 is available also — see `preprocess_data.py` for arguments.

</details>

---

<details>
<summary><b>RL training</b></summary>

### Launch training jobs
Each of the five agents from the paper has its own launch script under `scripts/train/experiments/nat-com-version/`:

| Agent | Reward terms | Script |
|-------|-------------|--------|
| A | Atomization energy | `a.py` |
| F | Formation energy | `f.py` |
| AV | Atomization energy + Validity | `av.py` |
| FV | Formation energy + Validity | `fv.py` |
| AFV | Atomization energy + Formation energy + Validity | `afv.py` |

Run any agent with:
```bash
python scripts/train/experiments/nat-com-version/a.py   # Agent A
python scripts/train/experiments/nat-com-version/afv.py # Agent AFV
# etc.
```

All training metrics and checkpoints are saved to `runs/<agent-name>/seed_<n>/` regardless of whether W&B is enabled.

### Weights & Biases (optional)
Training metrics can optionally be tracked with [Weights & Biases](https://wandb.ai/). To disable it, set `save_to_wandb=False` in the training script — all metrics will then be saved to disk only.

To enable W&B, log in and set your entity:
```bash
wandb login
export WANDB_ENTITY=your_wandb_username
```

### Monitoring training without W&B
Learning curves can be plotted directly from the on-disk logs (no W&B account needed):

```bash
# Single agent:
python scripts/analyse/plot_learning_curves.py runs/Agent-A

# All five agents side by side:
python scripts/analyse/plot_learning_curves.py runs/Agent-A runs/Agent-F runs/Agent-AV runs/Agent-FV runs/Agent-AFV

# Custom smoothing window and output path:
python scripts/analyse/plot_learning_curves.py runs/Agent-A --smooth 200 --out figs/curves.png --csv curves.csv
```

This plots validity rate, mean return, and atomization energy vs training steps, and can optionally export a CSV of the smoothed curves.

</details>

---

<details>
<summary><b>Analysis scripts (isomer discovery campaigns)</b></summary>

The `scripts/analyse/` folder contains analysis scripts organized by experiment. The three main ones corresponding to the paper's research questions are:

- **exp1:** Single-bag evaluation (Q1 in the paper).
- **exp2:** Multi-bag evaluations on hold-out formulas (Q2 in the paper).
- **exp3:** Cumulative discovery analysis where the entire training is seen as a discovery campaign (Q3 in the paper).

</details>

---

<details>
<summary><b>Web app</b></summary>

A Streamlit-based interface for interactive molecule generation. Launch it with:

```bash
streamlit run app_store/About.py
```

<img src="resources/web-app.png" width="100%">

See [`app_store/README.md`](app_store/README.md) for full usage instructions.

</details>

---

<details>
<summary><b>Known Issues and Planned Improvements</b></summary>

Contributions welcome.

### High priority

- **Confusing `config_ft` structure.** The online RL (PPO) parameters live inside `config["config_ft"]` — a name inherited from an earlier pretraining→finetuning workflow that was not used in the final paper. In the paper, all agents are trained from scratch (*tabula rasa*). The nesting and the name are confusing; we plan to flatten the config structure in a future refactor.


</details>
