import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px


def bootstrap_estimate(sample, estimator, num_bootstrap_samples=100):
    """
    Estimate using bootstrapping by repeatedly resampling and applying an estimator.

    Args:
        sample: Data to resample.
        estimator: Function to estimate a value from the resampled data.
        num_bootstrap_samples: Number of bootstrap samples to generate.

    Returns:
        list: Estimated values for each bootstrap sample.
    """
    values = []

    for _ in range(num_bootstrap_samples):
        bootstrap_sample = np.random.choice(sample, size=len(sample), replace=False)
        value = estimator(bootstrap_sample)
        values.append(value)

    return values


def estimate_expected_unique(samples):
    """
    Estimate the expected number of unique molecules for a given sample.

    Args:
        sample (list): List of SMILES strings representing the generated molecules.

    Returns:
        float: Estimated expected number of unique molecules for the given sample.
    """

    samples = [sample for sample in samples if sample is not None]

    def unique_count(bootstrap_sample):
        return len(set(bootstrap_sample))

    bootstrap_results = bootstrap_estimate(samples, unique_count)
    expected_value = np.mean(bootstrap_results)

    return expected_value


@st.cache_data
def bootstrap_expected_unique_curve(samples, num_bootstrap_samples=100):
    """
    Estimate the expected number of unique molecules as a function of sample size using bootstrapping.

    Args:
        samples (list of lists): List of lists, where each sublist represents a generated sample.
        num_bootstrap_samples (int): Number of bootstrap samples to generate.

    Returns:
        list: Estimated expected number of unique molecules for each sample size and model.
    """
    expected_values_per_model = []

    for model_sample in samples:
        model_expected_values = []

        for n in range(1, len(model_sample) + 1):
            estimator = lambda sample: estimate_expected_unique(sample[:n])  # noqa: E731
            bootstrap_results = bootstrap_estimate(model_sample, estimator, num_bootstrap_samples)
            expected_value = np.mean(bootstrap_results)
            model_expected_values.append(expected_value)

        expected_values_per_model.append(model_expected_values)

    return expected_values_per_model


def smooth_data(data, window_size=5):
    """Smooth data using a rolling average."""
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode="same")
    return smoothed_data


def calculate_slope(data):
    """
    Calculate the slope of a function using central difference and rolling averaging.

    Args:
        data (numpy.ndarray): Array containing function values at discrete indices.

    Returns:
        numpy.ndarray: Estimated smoothed slopes.
    """
    T = len(data)
    slopes = np.zeros(T)

    for t in range(T):
        if t > 0 and t < T - 1:
            slope = (data[t + 1] - data[t - 1]) / 2
            slopes[t] = slope

    return slopes


def jackknife_plot(df_stoch: pd.DataFrame):
    generated_molecules = df_stoch["SMILES"].values.tolist()
    y = bootstrap_expected_unique_curve([generated_molecules])

    cols = st.columns(2)
    with cols[0]:
        fig = px.line(
            x=np.arange(1, len(y[0]) + 1),
            y=y[0],
            labels={"x": "Sample Size", "y": "Expected Unique"},
        )

        window_size = 5
        remove_size = window_size // 2
        smoothed_y = smooth_data(y[0], window_size)
        smoothed_y[:remove_size] = [None] * remove_size
        smoothed_y[-remove_size:] = [None] * remove_size

        # add to previous figure
        fig.add_scatter(
            x=np.arange(1, len(smoothed_y) + 1),
            y=smoothed_y,
            mode="lines",
            name="Smoothed",
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=False)

    with cols[1]:
        # Marginal Gain of extra sample
        marginal_gain = calculate_slope(smoothed_y)

        # window_size // 2 are invalid from smoothing and 1 from slope calculation
        remove_size_slope = window_size // 2 + 1
        marginal_gain_final = marginal_gain[remove_size_slope:-remove_size_slope][-1]
        fig = px.line(
            x=np.arange(1, len(marginal_gain) + 1),
            y=marginal_gain,
            labels={"x": "Sample Size", "y": "Marginal Gain"},
        )
        fig.update_layout(
            title=dict(text=f"Marginal Gain of extra sample: {np.round(marginal_gain_final, 2)}", font=dict(size=22))
        )

        st.plotly_chart(fig, theme="streamlit", use_container_width=False)
