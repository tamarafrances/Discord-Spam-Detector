import streamlit as st
st.set_page_config(layout="wide", page_title='Exploratory Data Analysis', page_icon="🔎")





import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_excel('./Data/data-for-capstone.xlsx')
df.columns = df.columns.str.lower()
df = df.rename(columns={'spam?':'spam'})

df_notspam = df[df['spam'] == 'N']
df_spam = df[df['spam'] == 'Y']

#intro
st.title('Exploratory Data Analysis')
st.markdown("""---""")

#sample messages
st.subheader('Sample Messages')
col1, col2 = st.columns(2)

with col1:
    st.write("###")
    st.write('Not Spam')
    st.write(df_notspam['text'][0:20], use_container_width=True)
    st.write("#")

with col2:
    st.write("###")
    st.write('Spam')
    st.write(df_spam['text'][0:20], use_container_width=True)
    st.write("#")


#count of not spam vs. spam
st.subheader('Count of Not Spam (N) vs. Spam (Y)')
svns = px.bar(df['spam'].value_counts())
svns.update_layout(showlegend=False)
st.plotly_chart(svns, use_container_width=True)
st.write("#")

#most common words
st.subheader('Most Common Words')
cv = CountVectorizer()
df_cv = cv.fit_transform(df['text'])

words = pd.DataFrame(df_cv.A, columns=cv.get_feature_names_out())
wordcounts = pd.DataFrame(words.sum().sort_values(ascending=False).head(15))
wc_common = px.bar(wordcounts)
wc_common.update_layout(showlegend=False)
st.plotly_chart(wc_common, use_container_width=True)
st.write("#")

#calculating the wordcount of the text
wc = []
for string in df['text']:
    wc.append(len(string.strip().split(' ')))  
df['word_count'] = wc

#histogram for word count
st.subheader('Word Count')
fig = px.histogram(df, x='word_count', color='spam', barmode='overlay', nbins=50, title='Word Count of Not Spam vs. Spam')
st.plotly_chart(fig, use_container_width=True)
st.write("#")

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
st.write("#")

#spam vs not spam

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

from textblob import TextBlob
df['polarity'] = df['text'].apply(lambda x: TextBlob(x).polarity)
df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).subjectivity)

#Polarity
st.subheader('Polarity')
pol = px.bar(df['polarity'].groupby(df['spam']).mean())
pol.update_layout(showlegend=False)
st.plotly_chart(pol, use_container_width=True)
st.write("#")

#Subjectivity
st.subheader('Subjectivity')
subj = px.bar(df['subjectivity'].groupby(df['spam']).mean())
subj.update_layout(showlegend=False)
st.plotly_chart(subj, use_container_width=True)
st.write("#")
