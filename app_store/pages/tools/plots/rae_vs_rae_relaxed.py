import plotly.express as px
import pandas as pd
import streamlit as st


def plot_rae_vs_rae_relaxed(df_stoch: pd.DataFrame):
    fig = px.scatter(
        df_stoch,
        x="rae",
        y="rae_relaxed",
        color="rae",
        hover_data=["SMILES", "NEW_SMILES"],
        # size=sizes,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, theme="streamlit", use_container_width=False)
