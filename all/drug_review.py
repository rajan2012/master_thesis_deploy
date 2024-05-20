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

@st.cache_data
def load_data():
    # Load your DataFrame here
    return pd.read_csv("drug_review.csv")

# Load the data
df2 = load_data()


# Remove duplicate records based on 'Symptoms' and 'Disease' columns
df = df2.drop_duplicates(subset=['drug', 'Disease','rating','review'], keep='first')

# Strip leading and trailing whitespaces from the disease and symptoms columns
#df2= pd.DataFrame()

df.loc[:, 'Disease'] = df['Disease'].str.strip()
df.loc[:, 'drug'] = df['drug'].str.strip()
df.loc[:, 'Disease'] = df['Disease'].str.lower()

df_rating_count = df.groupby(['Disease', 'drug', 'rating']).size().reset_index(name='record_count')
#print(grouped_df)

#st.write(df_rating_count.head(10))

grouped = df_rating_count.groupby(['drug', 'Disease'])

#st.write(grouped.head(10))

# Create a Streamlit app
st.title("Drug Lookup")
# Group ratings by drug
#grouped = df_rating_count.groupby(['drug','Disease'])

# Remove duplicate records based on 'Symptoms' and 'Disease' columns
df3 = df.drop_duplicates(subset=['drug', 'Disease'], keep='first')
df3.loc[:, 'Disease'] = df3['Disease'].str.strip()
df3.loc[:, 'drug'] = df3['drug'].str.strip()
df3.loc[:, 'Disease'] = df3['Disease'].str.lower()
disease_list = df3['Disease'].unique()

# Prepend an empty string to the disease list
disease_list_with_empty = [''] + disease_list
# Dropdown menu to select the disease
selected_disease = st.selectbox("Select Disease:", disease_list_with_empty)

#st.write("Selected Disease:", selected_disease)

n = st.number_input("Enter the value of n:", min_value=0, step=1, value=40)


def calculate_weighted_avg_rating(df_rating_count, disease1,n):
    # Group ratings by drug
    grouped = df_rating_count.groupby(['drug','Disease'])

    #st.write(grouped.head(10))
    # Calculate weighted average rating for each drug
    weighted_avg_ratings = []
    for (drug, disease), group in grouped:
        weighted_sum = (group['rating'] * group['record_count']).sum()
        total_users = group['record_count'].sum()
        # Get the disease value from the first row of the group
        # disease = group['Disease'].iloc[0]  # Use this if 'Disease' is the column name
        # disease = disease  # Use this if 'Disease' is a separate variable
        # Adjust the weight by considering the total number of users
        weighted_avg = (weighted_sum + total_users) / (total_users + 1)

        weighted_avg_ratings.append({'Disease': disease, 'drug': drug, 'Weighted_Avg_Rating': weighted_avg, 'user_cnt': total_users})

    # Convert result to DataFrame
    result_df = pd.DataFrame(weighted_avg_ratings)
    #st.write(disease1)
    # Normalize ratings (optional)
    # Example: Normalize ratings to a scale of 0 to 10
    max_rating = result_df['Weighted_Avg_Rating'].max()
    min_rating = result_df['Weighted_Avg_Rating'].min()
    result_df['Normalized_Rating'] = 10 * (result_df['Weighted_Avg_Rating'] - min_rating) / (max_rating - min_rating)

    #st.write(result_df)

    # Filter the DataFrame to include only the top drugs for the specified disease
    top_drugs = result_df[result_df['Disease'] == disease1].sort_values(by='Normalized_Rating', ascending=False).head(n)
    #st.write(top_drugs)
    return top_drugs
result_df = calculate_weighted_avg_rating(df_rating_count, selected_disease, n)

if st.button("Submit"):
    #st.write(df_rating_count.head(2))
    #st.write(n)
    #st.write(selected_disease)
    # Call the function to calculate weighted average rating and return top drugs

    # Display rating and drug columns from result_df
    #st.write(result_df[['drug', 'Normalized_Rating']], index=False)
    st.write(result_df[['drug', 'Normalized_Rating']])

# Define thresholds for polarity categories
positive_threshold = 6
negative_threshold = 3

# Categorize polarity column into positive, neutral, and negative
def categorize_rating(rating):
    if rating >= positive_threshold:
        return 'Positive'
    elif rating <= negative_threshold:
        return 'Negative'
    else:
        return 'Neutral'

#no of user put positive ,negative review for top 10 popular drug
df['rating_category'] = df['rating'].apply(categorize_rating)

disease_drugs_df = df[(df['Disease'] == selected_disease) & (df['drug'].isin(result_df['drug']))]
def plot_stacked_bar_chart(df, top_10_drugs,disease):
    # Define colors for different review ratings
    colors = {'Negative': 'red', 'Neutral': 'orange','Positive': 'green'}

    # Filter the DataFrame to include only the top 15 drugs
    #top_15_drugs_df = df[df['Disease'] == disease]
    #top_15_drugs_df = top_15_drugs_df[top_15_drugs_df['drug'].isin(top_10_drugs['drug'])]


    # Group by 'drug' and 'rating_category' and count occurrences
    grouped_df = top_10_drugs.groupby(['drug', 'rating_category']).size().unstack(fill_value=0)

    # Plot stacked bar chart
    ax = grouped_df.plot(kind='bar', stacked=True, figsize=(12, 8), color=[colors.get(col, 'blue') for col in grouped_df.columns])

    # Add count annotations to each bar
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    # Add total count annotations at the top of each stacked bar
    for i, drug in enumerate(grouped_df.index):
        total_count = grouped_df.iloc[i].sum()
        ax.text(i, total_count , f'{total_count}', ha='center')

    plt.title('Distribution of Review Ratings for Top 15 Drugs')
    plt.xlabel('Drug')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Review Rating')
    #plt.show()
    # Display the plot in Streamlit
    st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

plot_stacked_bar_chart(df, disease_drugs_df,selected_disease)


def analyze_reviews(result_df):
    # Filter the DataFrame to include only the top N drugs for the specified disease
    #top_drugs = result_df[result_df['Disease'] == disease].sort_values(by='Normalized_Rating', ascending=False).head(n)


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
    result_df['processed_reviews'] = result_df['review'].apply(preprocess_text2)

    #st.write(result_df.head(10))
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
    #plt.show()
    st.pyplot()

# Call the method
analyze_reviews(disease_drugs_df)
