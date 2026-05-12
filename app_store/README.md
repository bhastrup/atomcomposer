# Web App

A Streamlit-based interface for interactive molecule generation with trained RL agents.

## Launch

Run from the project root (with `atomcomposer` activated):

```bash
streamlit run app_store/About.py
```

![Web app screenshot](../resources/web-app.png)

## Pages

### About
The landing page. Displays the paper citation and an overview of generated molecule examples.

---

### Generator
The main inference page, laid out in three columns.

#### Agents 🤖 (left column)
Five agents from the paper (A, AV, F, FV, AFV) are pre-loaded from `model_objects/` on startup (seed 0). To load a custom checkpoint, use the **Agent Loader** expander: either upload a file or paste a file path directly (e.g. `runs/Agent-A/seed_1/_steps-50000.model`), select the device (`cpu` or `cuda`), give the agent a name, and click **Load agent**. Additional seeds are available under `model_objects/`.

#### Environment 🌍 (right column)
Use the **Env Builder** expander to configure the chemical formula the agent will build. Two environment types are available:

- **From scratch**: The agent builds a molecule from an empty canvas. The formula (bag) can be set from several sources — select one of the five paper formulas (C3H8O, C4H7N, C3H5NO3, C7H10O2, C7H8N2O2), type a custom formula (e.g. `C3H5NO3`), or sample randomly from QM7.
- **Partial canvas**: The agent completes a partially built molecule. Load a molecule from the **Explore Datasets** page first, then specify which atoms to remove — the agent will attempt to rebuild the missing fragment.

Configured environments appear under **All envs** and can be assigned to a playground.

#### Generator 🚀 (center column)
1. Click **New Playground** to create a playground in edit mode.
2. Assign an agent (left column → **To Playgrounds**) and an environment (right column → **To Playgrounds**).
3. Click **Deploy** to lock in the pair.
4. Configure generation settings and click a generate button to run.

**Generation settings:**

| Setting | Description |
|---------|-------------|
| **Argmax** ⛰️ | Greedy (deterministic) rollout — one molecule, always the same |
| **Stochastic** 🎲 | Samples from the policy distribution — run many times for diversity |
| **Num samples** | Number of stochastic molecules to generate (1–10,000) |
| **Num workers** | Parallel environment workers (1–128) — increase for large sample counts |
| **Render** | Visualise atom placement step-by-step during generation |
| **With bonds** | Show bonds during rendering (slower) |
| **Relax** | Run xTB geometry optimisation on each generated molecule |
| **Dipole** | Compute the xTB dipole moment for each molecule |

Start small (≤ 50 samples) until you are familiar with the expected runtime.

---

### Results
Results appear in tabs below the generator, one tab per deployed playground. For each agent–environment pair you will see:

**Metrics** — reported for both argmax and stochastic rollouts:
- *Validity*: fraction of molecules that pass RDKit sanitisation
- *Uniqueness*: fraction of valid molecules with a distinct SMILES
- *Rediscovery / novelty*: overlap with and expansion beyond the reference dataset
- *Energetics*: mean atomisation energy, mean dipole, relaxation stability, RMSD, and ring statistics

**Visualisation** — per-molecule buttons on each result row:
- **View**: ASE 3D viewer
- **View Seq**: step-by-step atom placement replay
- **View RdKit**: 2D structural diagram
- **View Opt**: relaxed geometry (requires Relax enabled)

**Analysis plots** — optional checkboxes to render:
- Energy histogram
- RAE (relative atomisation energy) distribution
- Basin / RMSD scatter
- Jackknife uncertainty plot
- RAE vs RAE-relaxed comparison

Reference energies from QM7/QM9 can be loaded for direct comparison by ticking **Load reference data** in the results tab.

---

### Explore Datasets
Browse the QM7, QM9, or tmQM reference datasets. Use the sidebar to select a dataset, then filter molecules by atom count, element presence, or chemical formula. For any selected bag, view individual conformers, inspect energetics (absolute and relative atomisation energy), and explore the atom-placement decomposition schemes used during training. Molecules can be loaded into the **Partial canvas** environment builder for targeted generation experiments.
