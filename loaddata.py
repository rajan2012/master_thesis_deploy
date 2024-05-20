import streamlit as st

import pandas as pd

@st.cache_data
def load_data(filename):
    # Load your DataFrame here
    return pd.read_csv(filename)
