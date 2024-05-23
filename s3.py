import os
import boto3
import pandas as pd
import streamlit as st

# Load AWS credentials from Streamlit secrets
aws_default_region = st.secrets["aws"]["AWS_DEFAULT_REGION"]
aws_access_key_id = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]

# Set environment variables
os.environ["AWS_DEFAULT_REGION"] = aws_default_region
os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

# Create an S3 client
s3_client = boto3.client('s3')
response = s3_client.get_object(Bucket="test22-rajan", Key="df_rating_cnt.csv")

# Read the CSV file from the response
file = response["Body"]
data2 = pd.read_csv(file)

# Display the DataFrame in Streamlit
st.write(data2)
