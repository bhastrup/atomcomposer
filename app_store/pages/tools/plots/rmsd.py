import plotly.express as px
import pandas as pd
import streamlit as st


def plot_basin_RMSD(df_stoch: pd.DataFrame):
    fig = px.scatter(
        df_stoch,
        x="basin_distance",
        y="RMSD",
        color="abs_energy",
        hover_data=["SMILES", "NEW_SMILES"],
        # size=sizes,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=False)
