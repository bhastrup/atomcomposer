import sys

import streamlit as st

from PIL import Image

from app_store.pages.tools.images.show_logo import show_logo

st.set_page_config(layout="wide")


show_logo()

col1, col2 = st.columns([5, 10])
with col1:
    st.title("Isomer Discovery with RL")
    st.write(
        """
        Accompanying web app for the paper:  
        **Rediscovering Chemical Space from First Principles with Reinforcement Learning**
        *Bjarke Hastrup, François Cornet, Tejs Vegge, and Arghya Bhowmik*
        Published in *Nature Communications*.
        Preprint available on [Research Square](https://doi.org/10.21203/rs.3.rs-6900238/v1).
        """
    )
    with st.expander("Python environment details 🐍", expanded=False):
        st.write(f"python environment: {sys.executable}")
        st.write(f"python version: {sys.version}")

with col2:
    st.image(Image.open("resources/image_grid.png"))
