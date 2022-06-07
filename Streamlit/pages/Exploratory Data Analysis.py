import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_excel('/Users/tamarafrances/Downloads/data-for-capstone.xlsx')
df.columns = df.columns.str.lower()
df = df.rename(columns={'spam?':'spam'})


#intro
st.header('Exploratory Data Analysis')
svns = px.bar(df['spam'].value_counts(), title='Count of Not Spam (N) and Spam (Y) in Dataset')
svns.update_layout(showlegend=False)
st.plotly_chart(svns, use_container_width=True)

#most common words
cv = CountVectorizer()
df_cv = cv.fit_transform(df['text'])

words = pd.DataFrame(df_cv.A, columns=cv.get_feature_names_out())
wordcounts = pd.DataFrame(words.sum().sort_values(ascending=False).head(15))
wc_common = px.bar(wordcounts, title='Most Common Words in Dataset')
wc_common.update_layout(showlegend=False)
st.plotly_chart(wc_common, use_container_width=True)

#calculating the wordcount of the text
wc = []
for string in df['text']:
    wc.append(len(string.strip().split(' ')))  
df['word_count'] = wc

#histogram for word count
st.subheader('Word Count')
fig = px.histogram(df, x='word_count', color='spam', barmode='overlay', nbins=50, title='Word Count of Not Spam vs. Spam')
st.plotly_chart(fig, use_container_width=True)

#calculating the length of the text
length = []
for string in df['text']:
    a = string.strip()
    length.append(len(a))
df['text_length'] = length

#histogram for text length
st.subheader('Text Length')
fig2 = px.histogram(df, x='text_length', color='spam', barmode='overlay', nbins=50, title = 'Text Length of Not Spam vs. Spam')
st.plotly_chart(fig2, use_container_width=True)

#spam vs not spam

df_notspam = df[df['spam'] == 'N']
df_spam = df[df['spam'] == 'Y']

st.subheader('Most Common Bigrams and Trigrams')
col1, col2 = st.columns(2)

with col1:
#bigrams and trigrams - not spam
    from nltk.corpus import stopwords
    stoplist = stopwords.words('english')
    c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3))
    # matrix of ngrams
    ngrams = c_vec.fit_transform(df_notspam['text'])
    # count frequency of ngrams
    count_values = ngrams.toarray().sum(axis=0)
    # list of ngrams
    vocab = c_vec.vocabulary_
    df_ngram_notspam = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
                ).rename(columns={0: 'frequency', 1:'bigram/trigram'})

    fig3 = px.bar(df_ngram_notspam.head(10), x="frequency", y="bigram/trigram", title='Not Spam')
    st.plotly_chart(fig3, use_container_width=True)

with col2:
#bigrams and trigrams - spam
    from nltk.corpus import stopwords
    stoplist = stopwords.words('english')
    c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3))
    # matrix of ngrams
    ngrams = c_vec.fit_transform(df_spam['text'])
    # count frequency of ngrams
    count_values = ngrams.toarray().sum(axis=0)
    # list of ngrams
    vocab = c_vec.vocabulary_
    df_ngram_spam = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
                ).rename(columns={0: 'frequency', 1:'bigram/trigram'})

    fig4 = px.bar(df_ngram_spam.head(10), x="frequency", y="bigram/trigram", title='Spam')
    st.plotly_chart(fig4, use_container_width=True)