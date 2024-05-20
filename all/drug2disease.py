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
df.loc[:, 'drug'] = df['drug'].str.lower()
# Assuming df is your DataFrame
df = df.dropna(subset=['Disease'])
df = df[~df['Disease'].str.contains('comment helpful')]

# Display the dataframe without duplicate records
#print(df)


# Extract distinct list of diseases from the dataset
drug_list = df['drug'].unique()

# Create a Streamlit app
st.title("disease Lookup")

# Dropdown menu to select the disease
selected_drug = st.selectbox("Select drug:", drug_list)

# Submit button
if st.button("Submit"):
    with st.spinner("Filtering dataset..."):
        # Filter the dataset based on the selected drug
        filtered_df = df[df['drug'] == selected_drug]

        # Extract distinct list of diseases associated with the selected drug
        disease_list = filtered_df['Disease'].unique()

        # Display the distinct diseases in a tabular format
        st.write("Diseases for Drug:")
        st.write(pd.DataFrame(disease_list, columns=['Disease List']))



