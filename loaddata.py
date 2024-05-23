import os
import boto3
import pandas as pd
import streamlit as st
import joblib
import pickle

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


def load_data_s3(bucket_name, file_key):
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

    # Get the object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)

    # Read the CSV file from the response
    file = response["Body"]
    data = pd.read_csv(file)

    return data




def load_pkl_s3(bucket_name, file_key):
    # Load AWS credentials from Streamlit secrets
    aws_default_region = st.secrets["aws"]["AWS_DEFAULT_REGION"]
    aws_access_key_id = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]

    # Set environment variables
    os.environ["AWS_DEFAULT_REGION"] = aws_default_region
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    # Create an S3 client
    #s3_client = boto3.client('s3')

    s3 = boto3.resource('s3')
    my_pickle = pickle.loads(s3.Bucket(bucket_name).Object(file_key).get()['Body'].read())

    return my_pickle
