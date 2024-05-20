import streamlit as st
import pandas as pd
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


import pickle

import joblib

#from predictdis import predict_disease
from preprocess import preprocess_text

# Add the image as a header
st.image("disease_predictor.png", use_column_width=True)

@st.cache_data
def load_data():
    # Load your DataFrame here
    return pd.read_csv("removedlongsym.csv")

# Load the data
df2 = load_data()


# Strip leading and trailing whitespaces from the 'Disease' and 'Symptoms' columns
df2.loc[:, 'Disease'] = df2['Disease'].str.strip()
df2.loc[:, 'Symptoms'] = df2['Symptoms'].str.strip()

# Modify 'Disease' column to lowercase
df2.loc[:, 'Disease'] = df2['Disease'].str.lower()

df2.loc[:, 'Disease'] = df2['Disease'].str.lower()
df2.loc[:, 'Symptoms'] = df2['Symptoms'].str.replace('_', ' ')
df2.loc[:, 'Symptoms'] = df2['Symptoms'].str.lower()


# Drop duplicate records based on 'Disease' and 'Symptoms' columns
df3 = df2.drop_duplicates(subset=['Disease', 'Symptoms'])

# Reset the index
df3.reset_index(drop=True, inplace=True)

# Count the occurrences of each disease
disease_counts = df3['Disease'].value_counts()

# Filter the DataFrame to include only diseases that occur more than once
df = df3[df3['Disease'].isin(disease_counts[disease_counts > 3].index)]

# Display the filtered DataFrame
#print(df_filtered)



# Extract distinct list of symptoms from the dataset
symptoms_list = df['Symptoms'].str.split(',').explode().unique()

# Title for the app
st.title('Symptom Selector')

# Create a multi-select box for selecting symptoms
selected_symptoms = st.multiselect('Select Symptoms:', symptoms_list)

# Apply preprocessing to the selected symptoms

user_symptoms = ','.join(selected_symptoms)


# Load the trained pipeline from the pickle file
#pipeline = joblib.load('RandomForest.pkl')

pipeline = joblib.load('RandomForest_new.pkl')
vectorizer1 = joblib.load('CountVectorizer_random.pkl')
#vectorizer = CountVectorizer(binary=True)
# Function to predict disease based on symptoms
def predict_disease(symptoms):
    # Preprocess symptoms
    processed_symptoms = preprocess_text(symptoms)
    st.write(processed_symptoms)
    # Predict disease using the trained pipeline
    #while using count vectorizer
    user_symptoms_vectorized = vectorizer1.transform([processed_symptoms])
    st.write(user_symptoms_vectorized)
    predicted_disease = pipeline.predict(user_symptoms_vectorized)

    return predicted_disease[0]

# Predict disease based on selected symptoms when button is clicked
if st.button('Predict Disease'):
    #st.write('Input Symptoms:', selected_symptoms)
    #st.write(user_symptoms)
    predicted_disease = predict_disease(user_symptoms)
    st.write("Predicted disease:", predicted_disease)

selected_options = []





#user_symptoms = "transaminitis, splenomegaly, apyrexial, lesion, cough, monoclonal, hypocalcemia result"
#print(type(user_symptoms))


