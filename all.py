import streamlit as st
import pandas as pd

from diseaseprediction import setup_and_run_symptom_selector
from drug2disease import filter_diseases_by_drug
from drug_disease import setup_and_run_drug_lookup
from drug_review import setup_and_run_drug_review
from predictinsrance import insurance


# Function to display the disease prediction page
drugfile="https://github.com/rajan2012/master_thesis_deploy/blob/main/drug_disease_unique.csv"
drugreview="https://github.com/rajan2012/master_thesis_deploy/blob/main/process_drug_reviews.csv"
diseaefile="https://github.com/rajan2012/master_thesis_deploy/blob/main/removedlongsym.csv"
#healthfile="https://github.com/rajan2012/master_thesis_deploy/blob/main/insurance_dataset.csv"
healthfile="https://github.com/rajan2012/master_thesis_deploy/blob/main/process_healthinsurance.csv"
# #"allcombo_hl.csv"
#from drug disease
diseaselist="https://github.com/rajan2012/master_thesis_deploy/blob/main/uniquedisease_drug_disease.csv"
#from drug reviews
reviewdiseaselist="https://github.com/rajan2012/master_thesis_deploy/blob/main/uniqdis_drug_rev.csv"
#with normlaizeed rating
normalizedrating="https://github.com/rajan2012/master_thesis_deploy/blob/main/normalized_rating.csv"
#groupby rating count
ratingcount="https://github.com/rajan2012/master_thesis_deploy/blob/main/df_rating_cnt.csv"
druglist="https://github.com/rajan2012/master_thesis_deploy/blob/main/uniquedrug.csv"


# File paths for the trained pipeline and vectorizer (replace with actual paths)
pipeline_path = 'https://github.com/rajan2012/master_thesis_deploy/blob/main/RandomForest_new.pkl'
vectorizer_path = 'https://github.com/rajan2012/master_thesis_deploy/blob/main/CountVectorizer_random.pkl'

def disease_prediction_page():
    st.title("Disease Prediction")
    st.write("This is the main page for disease prediction.")
    setup_and_run_symptom_selector(diseaefile,pipeline_path,vectorizer_path)
    # Add your disease prediction code here

# Function to display the drug discovery page
def drug_discovery_page():
    setup_and_run_drug_lookup(drugfile,diseaselist)
    # Add your drug discovery code here

# Function to display the drug reviews page
def drug_reviews_page():
    st.title("Drug Reviews")
    st.write("This is the drug reviews page.")
    setup_and_run_drug_review(drugreview,reviewdiseaselist,normalizedrating,ratingcount)
    # Add your drug reviews code here

def drug_disease_page():
    filter_diseases_by_drug(drugfile,druglist)
    # Add your drug reviews code here

# Function to display the health insurance page
def health_insurance_page():
    st.title("Health Insurance")
    st.write("This is the health insurance page.")
    insurance(healthfile)
    # Add your health insurance code here

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("health", ["Disease Prediction", "Drug Discovery","Drug Disease", "Drug Reviews", "Health Insurance"])

# Main content
if page == "Disease Prediction":
    disease_prediction_page()
elif page == "Drug Discovery":
    drug_discovery_page()
elif page == "Drug Disease":
    drug_disease_page()
elif page == "Drug Reviews":
    drug_reviews_page()
elif page == "Health Insurance":
    health_insurance_page()

