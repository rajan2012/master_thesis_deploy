from loaddata import load_data
import streamlit as st

filename="s3://masterthesisrajan/uniquedisease_drug_disease.csv"
df = load_data(filename)

st.write(df)