import streamlit as st

import pandas as pd

import requests
from io import StringIO

import streamlit as st
from st_files_connection import FilesConnection


@st.cache_data
def load_data_old(filename):
    # Load your DataFrame here
    return pd.read_csv(filename)

#for git file
def load_data(filename):
    conn = st.connection('s3', type=FilesConnection)
    df = conn.read(filename, input_format="csv", ttl=600)
    return df


# Create connection object and retrieve file contents.
# Specify input format is a csv and to cache the result for 600 seconds.
