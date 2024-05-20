import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    # Load your DataFrame here
    return pd.read_csv("drug_disease.csv")

# Load the data
df2 = load_data()


# Remove duplicate records based on 'Symptoms' and 'Disease' columns
df = df2.drop_duplicates(subset=['drug', 'Disease'], keep='first')

# Strip leading and trailing whitespaces from the disease and symptoms columns
# Strip leading and trailing whitespaces from the 'Disease' and 'Symptoms' columns
df.loc[:, 'Disease'] = df['Disease'].str.strip()
df.loc[:, 'drug'] = df['drug'].str.strip()

# Modify 'Disease' column to lowercase
df.loc[:, 'Disease'] = df['Disease'].str.lower()
# Display the dataframe without duplicate records
#print(df)


# Extract distinct list of diseases from the dataset
disease_list = df['Disease'].unique()

# Create a Streamlit app
st.title("Drug Lookup")

# Dropdown menu to select the disease
selected_disease = st.selectbox("Select Disease:", disease_list)

# Submit button
if st.button("Submit"):
    # Filter the dataset based on the selected disease
    filtered_df = df[df['Disease']==selected_disease]

    # Extract distinct list of drugs associated with the selected disease
    drug_list = filtered_df['drug'].unique()

    # Display the distinct drugs in a tabular format
    st.write("Distinct Drugs for Selected Disease:")
    st.write(pd.DataFrame(drug_list, columns=['Drug']))


