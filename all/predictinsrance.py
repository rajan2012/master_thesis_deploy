import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
#from predict_medical_cost import predict_medical_costs
from sklearn.preprocessing import LabelEncoder

# Assuming the file is named "your_file.csv" and is located in the current directory
#file_path = "insurance_dataset.csv"

@st.cache_data
def load_data():
    # Load your DataFrame here
    return pd.read_csv("insurance_dataset.csv")

# Load the data
df2 = load_data()
# Read the CSV file into a DataFrame   ,nrows=10
#df2 = pd.read_csv(file_path)

df=df2.head(500)
# Assuming 'df' is your DataFrame
df['medical_history'].fillna('None', inplace=True)
# Assuming 'df' is your DataFrame
df['family_medical_history'].fillna('None', inplace=True)

# Initialize LabelEncoder
label_encoder = LabelEncoder()
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
# Fit label encoder on all categorical columns
label_encoder.fit(pd.concat([df['medical_history'], df['exercise_frequency'], df['occupation'], df['coverage_level'], df['family_medical_history']]))


def preprocess_dataframe_user(df):
    # Initialize LabelEncoder

    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    #print(df)
    # Apply mapping functions to respective columns
    #df['medical_history'] = df['medical_history'].apply(map_medical)
    #df['exercise_frequency'] = df['exercise_frequency'].apply(map_exercise)
    df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
    #df['family_medical_history'] = df['family_medical_history'].apply(map_medical)
    #df['bmi'] = df['bmi'].apply(map_bmi)
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'male' else 0)
    #df['age1'] = df['age'].apply(map_age)


    df['exercise_encoded'] = label_encoder.transform(df['exercise_frequency'])


    df['occupation_encoded'] = label_encoder.transform(df['occupation'])


    df['coverage_level_encoded'] = label_encoder.transform(df['coverage_level'])


    df['medical_history_encoded'] = label_encoder.transform(df['medical_history'])

    df['family_medical_history_encoded'] = label_encoder.transform(df['family_medical_history'])

    return df


def calculate_bmi(weight_kg, height):
    """
    Calculate BMI (Body Mass Index) given weight in kilograms and height in meters.
    """
    height_m=height/100
    bmi = weight_kg / (height_m ** 2)
    return bmi


# Function to predict medical costs based on user input
def predict_medical_costs(user_input):
    # Load the trained models
    #random_forest_model = joblib.load('Random Forest_model.pkl')
    xgboost_model = joblib.load('XGBoost_model.pkl')
    #linear_regression_model = joblib.load('LinearRegression_model.pkl')

    # Prepare input data as a DataFrame
    # Convert dictionary to DataFrame
    # Convert dictionary to DataFrame
    user_df = pd.DataFrame.from_dict(user_input, orient='index').T
    #print("before transform",user_df)
    user_df = preprocess_dataframe_user(user_df).drop(columns=['medical_history','family_medical_history','coverage_level','exercise_frequency','occupation'],axis=1)
    #print("after transform",user_df)

    # Predict medical costs using each model
    #rf_prediction = random_forest_model.predict(user_df)
    xgb_prediction = xgboost_model.predict(user_df)
    #lr_prediction = linear_regression_model.predict(user_df)

    # Return predictions
    return {
        #'Random Forest': rf_prediction[0],
        'XGBoost': xgb_prediction[0],
        #'Linear Regression': lr_prediction[0]
    }

# Get user input for features
age = st.number_input("Enter age:", min_value=0, step=1)
gender = st.radio("Enter gender:", options=['male', 'female'])
weight = st.number_input("Enter weight (kg):", min_value=0.0, step=1.0)
height = st.number_input("Enter height (cm):", min_value=0, step=1)
# Calculate BMI
bmi = calculate_bmi(weight, height)
children = st.number_input("Enter number of children:", min_value=0, step=1)
smoker = st.radio("Enter smoker status:", options=['yes', 'no'])
medical_history = st.selectbox("Enter medical history:", options=['Diabetes', 'None', 'High blood pressure','heart disease'])
family_medical_history = st.selectbox("Enter family medical history:",
                                      options=['Diabetes', 'None', 'High blood pressure','heart disease'])
exercise_frequency = st.selectbox("Enter exercise frequency:",
                                  options=['Never', 'Occasionally', 'Rarely', 'Frequently'])
occupation = st.selectbox("Enter occupation:", options=['Blue collar', 'White collar', 'Student', 'Unemployed'])
coverage_level = st.selectbox("Enter coverage level:", options=['Premium', 'Standard', 'Basic'])

# Add a button to trigger prediction
if st.button("Predict Cost"):
    # Collect user input
    user_input = {
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'medical_history': medical_history,
        'family_medical_history': family_medical_history,
        'exercise_frequency': exercise_frequency,
        'occupation': occupation,
        'coverage_level': coverage_level,
    }

    # Make predictions
    predictions = predict_medical_costs(user_input)

    # Display predictions
    st.header("Predicted Medical Costs:")
    for model, cost in predictions.items():
        st.write(f"{model}: €{cost:.2f}")

