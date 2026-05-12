import streamlit as st
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff


def get_bin_size(valid_energies, invalid_energies) -> float:
    if valid_energies.size > 0:
        min_valid = min(valid_energies)
        max_valid = max(valid_energies)
    else:
        min_valid = float("inf")
        max_valid = float("-inf")

    if invalid_energies.size > 0:
        min_invalid = min(invalid_energies)
        max_invalid = max(invalid_energies)
    else:
        min_invalid = float("inf")
        max_invalid = float("-inf")

    min_val = min(min_valid, min_invalid)
    max_val = max(max_valid, max_invalid)

    width = max_val - min_val
    bin_size = width / 50

    return bin_size


def energy_histogram(df) -> None:
    """Plot the energy histogram of a dataframe"""

    df["valid"] = df["valid"].astype(bool)

    # Filter data for valid and invalid categories
    valid_energies = df[df["valid"]]["abs_energy"].values
    invalid_energies = df[~df["valid"]]["abs_energy"].values

    if len(valid_energies) == 0 and len(invalid_energies) == 0:
        st.warning("No data to display for either 'Valid' or 'Invalid' category.")
        return

    if len(valid_energies) == 1 or len(invalid_energies) == 1:
        st.warning("Only one data point in either 'Valid' or 'Invalid' category. Cannot create histogram.")
        return

    bin_size = get_bin_size(valid_energies, invalid_energies)

    if len(valid_energies) > 0 and len(invalid_energies) > 0:
        hist_data = [valid_energies, invalid_energies]
        label = ["Valid", "Invalid"]
        bins = [bin_size, bin_size]
    else:
        bins = [bin_size]
        if len(valid_energies) > 0:
            hist_data = [valid_energies]
            label = ["Valid"]
        elif len(invalid_energies) > 0:
            hist_data = [invalid_energies]
            label = ["Invalid"]

    fig = ff.create_distplot(hist_data, label, bin_size=bins)
    st.plotly_chart(fig, theme="streamlit", use_container_width=False)


def energy_histogram_numpy(df, column_name="abs_energy"):
    """Plot the energy histogram of a dataframe"""

    # only use rows where column_name is not null/None/NaN
    df = df.dropna(subset=[column_name])

    df["valid"] = df["valid"].astype(bool)

    # Filter data for valid and invalid categories
    valid_energies = df[df["valid"]][column_name].values
    invalid_energies = df[~df["valid"]][column_name].values

    # Calculate the bin edges using np.linspace
    energy_range = (min(df[column_name]), max(df[column_name]))

    # num_bins = int((energy_range[1] - energy_range[0]) / bin_size) + 1
    # energy_unit = 'eV'
    # if energy_unit == 'Hartree':
    #     width = 0.007
    #     bin_size = 0.01
    # elif energy_unit == 'eV':
    #     width = 0.15
    #     bin_size = 0.23

    num_bins = 50
    bins = np.linspace(energy_range[0], energy_range[1], num_bins)
    bin_size = bins[1] - bins[0]
    width = bin_size * 0.75
    opacity = 0.5

    # Check if there's data in each category before creating the plot
    if len(valid_energies) > 0 and len(invalid_energies) > 0:
        # Create histograms manually for both categories
        valid_hist, valid_bins = np.histogram(valid_energies, bins=bins)
        invalid_hist, invalid_bins = np.histogram(invalid_energies, bins=bins)

        # Create a custom plot with Plotly
        trace1 = go.Bar(x=valid_bins, y=valid_hist, name="Valid", opacity=1, width=width, marker=dict(color="navy"))
        trace2 = go.Bar(
            x=invalid_bins, y=invalid_hist, name="Invalid", opacity=opacity, width=width, marker=dict(color="orange")
        )

        layout = go.Layout(title="Energy Histogram", xaxis=dict(title="Energy"), yaxis=dict(title="Counts"))

        data = [trace1, trace2]
        fig = go.Figure(data=data, layout=layout)

        # Adjust bargap and bargroupgap to control the bar width
        fig.update_layout(bargap=0.1, bargroupgap=0.1)

        st.plotly_chart(fig, use_container_width=False)
    else:
        if len(valid_energies) > 0:
            # Create a histogram for the 'Valid' category
            valid_hist, valid_bins = np.histogram(valid_energies, bins=bins)
            trace1 = go.Bar(
                x=valid_bins, y=valid_hist, name="Valid", opacity=opacity, width=width, marker=dict(color="blue")
            )
            layout = go.Layout(title="Valid Energy Histogram", xaxis=dict(title="Energy"), yaxis=dict(title="Counts"))
            fig = go.Figure(data=[trace1], layout=layout)

            # Adjust bargap and bargroupgap to control the bar width
            fig.update_layout(bargap=0.2, bargroupgap=0.1)

            st.plotly_chart(fig, use_container_width=False)

        if len(invalid_energies) > 0:
            # Create a histogram for the 'Invalid' category
            invalid_hist, invalid_bins = np.histogram(invalid_energies, bins=bins)
            trace2 = go.Bar(
                x=invalid_bins,
                y=invalid_hist,
                name="Invalid",
                opacity=opacity,
                width=width,
                marker=dict(color="orange"),
            )
            layout = go.Layout(title="Invalid Energy Histogram", xaxis=dict(title="Energy"), yaxis=dict(title="Counts"))
            fig = go.Figure(data=[trace2], layout=layout)

            # Adjust bargap and bargroupgap to control the bar width
            fig.update_layout(bargap=0.2, bargroupgap=0.1)

            st.plotly_chart(fig, use_container_width=False)

        if len(valid_energies) == 0 and len(invalid_energies) == 0:
            st.warning("No data to display for either 'Valid' or 'Invalid' category.")
