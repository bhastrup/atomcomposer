import logging
from typing import Dict

import numpy as np
import pandas as pd


def make_serializable(d: dict) -> dict:
    """Avoid int64 to make it JSON serializable"""
    for k in d:
        if isinstance(d[k], np.int64):
            d[k] = int(d[k])
    return d


def single_formula_metrics(df: pd.DataFrame, SMILES_db: list = None) -> dict:
    if len(df) == 0:
        return {}

    # drop rows where 'abs_energy' is None or nan
    df = df.dropna(subset=["abs_energy"])

    agg_metrics = {}
    new_set = set(df["SMILES"].dropna().unique())
    n_unique = len(new_set)
    total_samples = len(df)
    agg_metrics["total_samples"] = total_samples

    # Rediscovery and expansion metrics
    if SMILES_db is not None:
        assert isinstance(SMILES_db, list) and len(SMILES_db) > 0, "SMILES_db must be a list of SMILES strings"
        old_set = set([x for x in SMILES_db if str(x) != "nan"])
        num_old = len(old_set)

        novel_set = new_set - old_set
        n_novel = len(novel_set)
        rediscovered_set = new_set.intersection(old_set)
        n_rediscovered = len(rediscovered_set)

        rediscovery_ratio = len(rediscovered_set) / num_old
        expansion_ratio = len(new_set - old_set) / num_old
    else:
        num_old = None
        n_novel = None
        n_rediscovered = None
        rediscovery_ratio = None
        expansion_ratio = None

    agg_metrics["old_data_size"] = num_old
    agg_metrics["rediscovered"] = n_rediscovered
    agg_metrics["n_novel"] = n_novel
    agg_metrics["rediscovery_ratio"] = rediscovery_ratio
    agg_metrics["expansion_ratio"] = expansion_ratio

    # Validity ratio
    n_valid = df["valid"].sum()
    agg_metrics["n_valid"] = n_valid
    agg_metrics["valid_per_sample"] = n_valid / total_samples if total_samples > 0 else 0

    # Uniqueness measures
    agg_metrics["n_unique"] = n_unique
    agg_metrics["unique_per_sample"] = n_unique / total_samples if total_samples > 0 else 0
    agg_metrics["unique_per_valid"] = n_unique / n_valid if n_valid > 0 else 0

    # Energy measures
    agg_metrics["abs_energy_avg"] = df["abs_energy"].mean()
    agg_metrics["dipole_avg"] = df["dipole"].mean()

    # After relaxation measures
    agg_metrics["relax_stable"] = np.sum(df["relax_stable"]) / total_samples if total_samples > 0 else 0
    agg_metrics["RMSD_avg"] = df["RMSD"].mean()

    # RAE measures
    rae_relaxed = df["rae_relaxed"].dropna()
    agg_metrics["rae_relaxed_avg"] = rae_relaxed.mean() if len(rae_relaxed) > 0 else None

    # Ring measures
    agg_metrics["n_rings_avg"] = df["n_rings"].mean()

    agg_metrics["ring3+_ratio"] = sum(df["n_atoms_ring_max"] >= 3) / total_samples if total_samples > 0 else 0
    agg_metrics["ring4+_ratio"] = sum(df["n_atoms_ring_max"] >= 4) / total_samples if total_samples > 0 else 0
    agg_metrics["ring5+_ratio"] = sum(df["n_atoms_ring_max"] >= 5) / total_samples if total_samples > 0 else 0

    agg_metrics = make_serializable(agg_metrics)

    return agg_metrics


def get_global_metrics(formula_metrics: Dict[str, Dict], size_weighted: bool = True) -> dict:
    """Calculate the global metrics from a dictionary of formula metrics"""
    # INFO: To keep the weights from fluctuating we use the desired ('old_data_size')
    # rather than realized number of samples ('total_samples') for the weighting

    formulas = list(formula_metrics.keys())

    # Find any formulas that don't have all the metric keys and eject them
    metric_names = set().union(*[formula_metrics[formula].keys() for formula in formulas])
    missing_keys = []
    for formula in formulas:
        if not all(key in formula_metrics[formula] for key in metric_names):
            missing_keys.append(formula)
            logging.info(
                f"Warning: Formula {formula} is missing metrics: "
                + f"{[key for key in metric_names if key not in formula_metrics[formula]]}"
            )
    formulas = [f for f in formulas if f not in missing_keys]

    mean_agg_metrics = {}
    if size_weighted:
        assert "old_data_size" in metric_names, "old_data_size must be a metric"
        weights = {formula: formula_metrics[formula]["old_data_size"] for formula in formulas}
    else:
        weights = {formula: 1.0 for formula in formulas}

    weights = {formula: weight / sum(weights.values()) for formula, weight in weights.items()}

    for metric in metric_names:
        mean_agg_metrics[metric] = sum(
            [
                formula_metrics[formula][metric] * weights[formula]
                if metric in formula_metrics[formula] and formula_metrics[formula][metric] is not None
                else 0
                for formula in formulas
            ]
        )

    return mean_agg_metrics
