import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


def plot_stacked_bar_chart2(top_10_drugs):
    df = top_10_drugs.groupby(['drug', 'rating_category']).size().unstack(fill_value=0)

    # Reset index to make 'drug' a separate column
    df = df.reset_index()
    #st.write(df)

    # Calculate total count for each drug
    df['Total'] = df[['Positive', 'Negative', 'Neutral']].sum(axis=1)

    # Melt the DataFrame for Plotly
    df_melted = df.melt(id_vars=['drug', 'Total'], value_vars=['Positive', 'Negative', 'Neutral'], var_name='Sentiment',
                        value_name='Count')

    # Plot with Plotly Express
    fig = px.bar(df_melted, x='Count', y='drug', color='Sentiment', orientation='h',
                 height=800, width=1000, title='Sentiment Counts by Drug',
                 color_discrete_map={'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'},
                 labels={'Count': 'Count per Sentiment'})

    # Add total count annotations
    for i, row in df.iterrows():
        fig.add_annotation(x=row['Total'], y=row['drug'], text=f'{row["Total"]}', showarrow=False, font=dict(color='black', size=12))

    # Update layout for better spacing and legend background color
    fig.update_layout(barmode='stack', yaxis={'categoryorder': 'total ascending'},
                      legend=dict(bgcolor='black', bordercolor='black', borderwidth=1))

    # Display the plot
    st.plotly_chart(fig)