import pandas as pd
import streamlit as st
import gensim
from gensim import corpora
from gensim.models import LdaModel
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import scipy
from wordcloud import WordCloud
from collections.abc import Mapping
import plotly.express as px
import pandas as pd

import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from loaddata import load_data, load_data_s3

from plotdrug import  plot_stacked_bar_chart2,plot_stacked_bar_chart_3,plot_stacked_bar_chartavg


def analyze_reviews_drug_new(df, drug):
    # Filter the DataFrame to include only the top N drugs for the specified disease
    disease_drugs_df = df[df['drug'] == drug]

    def preprocess_text2(text):
        # Tokenization
        tokens = word_tokenize(text.lower())

        # Remove punctuation and stopwords
        stop_words = set(stopwords.words('english') + list(string.punctuation))
        tokens = [token for token in tokens if token not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    # Preprocess the reviews
    disease_drugs_df['processed_reviews'] = disease_drugs_df['review'].apply(preprocess_text2)

    # Create a dictionary mapping of words to their integer ids
    dictionary = corpora.Dictionary(disease_drugs_df['processed_reviews'])

    # Convert the reviews into bag of words representation
    bow_corpus = [dictionary.doc2bow(review) for review in disease_drugs_df['processed_reviews']]

    # Train LDA model
    lda_model = LdaModel(bow_corpus, num_topics=20, id2word=dictionary, passes=10)

    # Print the topics and associated words
    #for topic_id, topic_words in lda_model.print_topics():
        #print(f"Topic {topic_id}: {topic_words}")

    # Initialize an empty list to store all words
    all_words = []

    # Iterate over each topic
    for topic_id in range(lda_model.num_topics):
        # Get the words associated with the current topic
        topic_words = lda_model.show_topic(topic_id, topn=10)  # Adjust topn as needed
        # Extract words and append to the list
        words = [word for word, _ in topic_words]
        all_words.extend(words)

    # Initialize a word frequency dictionary
    word_freq = {}

    # Count the frequency of each word
    for word in all_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    # Display the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def topndrugs(df, disease1,n):
    df3=df[['drug','rating_avg','rating_category_sentiment']]
    df3=df3.rename(columns={'rating_avg':'rating','rating_category_sentiment':'rating_category'})
    # Filter the DataFrame to include only the top drugs for the specified disease
    top_drugs = df3[df3['Disease'] == disease1].sort_values(by='rating', ascending=False).head(n)
    #st.write(top_drugs)
    return top_drugs

#setup_and_run_drug_review_new(bucket_name,drugreview,reviewdiseaselist,normalizedrating,ratingcount,avgratin)

def setup_and_run_drug_review_new(bucket_name,filename,filename2,filename3,filename4,avgrating,druglist):
    # Load the data
    df = load_data_s3(bucket_name, filename)
    normal_rating_df = load_data_s3(bucket_name,filename3)
    df_rating_count = load_data_s3(bucket_name,filename4)
    #df, df_rating_count = preprocess_and_group_data(df)
    #unique disease
    unique_dis_df = load_data_s3(bucket_name,filename2)

    avgrat_df = load_data_s3(bucket_name,avgrating)

    uniq_drug = load_data_s3(bucket_name,druglist)

    # Remove duplicate records based on 'drug' and 'Disease' columns
    #df3 = df.drop_duplicates(subset=['drug', 'Disease'], keep='first')
    #df3.loc[:, 'Disease'] = df3['Disease'].str.strip().str.lower()
    #df3.loc[:, 'drug'] = df3['drug'].str.strip()

    # Extract distinct list of diseases from the dataset
    #disease_list = df3['Disease'].unique()
    disease_list = unique_dis_df['Disease']

    # Prepend an empty string to the disease list
    disease_list_with_empty = [''] + list(disease_list)

    drug_list = uniq_drug['drug']

    # Prepend an empty string to the disease list
    drug_list_with_empty = [''] + list(drug_list)

    with st.form(key='user_input_form2'):
        # Dropdown menu to select the disease
        selected_disease = st.selectbox("Select Medical Condition :", disease_list_with_empty)

        # Number input for the value of n
        n = st.number_input("Enter the value of n:", min_value=0, step=1, value=40)

        # Submit button
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        #this will also go away
        #result_df = calculate_weighted_avg_rating(df_rating_count, selected_disease, n)
        result_df = topndrugs(normal_rating_df, selected_disease, n)

        #st.write(result_df[['drug', 'Normalized_Rating']], index=False)

        # Assuming result_df is your DataFrame
        #result_df_subset = result_df[['drug', 'user_cnt', 'Rating']]
        # Assuming result_df_subset is your DataFrame
        #result_df_subset['Normalized_Rating'] = result_df_subset['Rating'].round(2)
        # Display the DataFrame in table format without index
        st.write(result_df)
        # Assuming result_df_subset is your DataFrame
        #result_df_subset_html = result_df_subset.to_html(index=False)

        # Display the DataFrame in table format without index and row numbers
        #st.write(result_df_subset_html, unsafe_allow_html=True)

        # Categorize ratings
        #this also in modified df , modified df
        #df['rating_category'] = df['rating'].apply(categorize_rating)
        #get all records for selected disease and drug which is in top n 
        #disease_drugs_df = df[(df['Disease'] == selected_disease) & (df['drug'].isin(result_df['drug']))]

        disease_drugs_df2 = avgrat_df[(avgrat_df['Disease'] == selected_disease) & (avgrat_df['drug'].isin(result_df['drug']))]

        #rename
        disease_drugs_df=disease_drugs_df2.rename(columns={'avg_rating':'rating'})

        # Call the method 
        #analyze_reviews_new only for selected sepecifc drug with diff submit button 
        #analyze_reviews_new(disease_drugs_df)
        #diseas_drug_df - contains disease drug for which disease has been choosed
        #for the df passed in plot we don't need comment by user
        #only rating_category,disease,drug

        #use avgrating_drug_29thnov file for visualizing on barchart
        disease_drugs_df_sub = disease_drugs_df[['drug', 'Disease', 'rating_category']]
        #plot_stacked_bar_chart(disease_drugs_df_sub)
        grouped_df = disease_drugs_df_sub.groupby(['drug', 'rating_category']).size().reset_index(name='counts')
        #plot_review_distribution_new(disease_drugs_df_sub)
        plot_stacked_bar_chartavg(avgrat_df,disease_drugs_df_sub,selected_disease)
        #st.write(grouped_df)


##for wordcloud for each selected drug 
#also bar chart for selected drug 
    with st.form(key='user_input_form'):
        # Dropdown menu to select the disease
        selected_drug = st.selectbox("Select drug :", drug_list_with_empty)

        # Submit button
        submit_button_wordcloud = st.form_submit_button(label="wordcloud")
        submit_button_bar = st.form_submit_button(label="barchart")

    if submit_button_wordcloud:
        #this will also go away
        #result_df = calculate_weighted_avg_rating(df_rating_count, selected_disease, n)
        #get all record with selected_drug
        #df is with processed reviews
        analyze_reviews_drug_new(df,selected_drug)

    if submit_button_bar:
        #this will also go away
        #result_df = calculate_weighted_avg_rating(df_rating_count, selected_disease, n)
        #get all record with selected_drug
        #
        plot_stacked_bar_chart_3(avgrat_df,disease_drugs_df_sub,selected_drug)