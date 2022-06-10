import streamlit as st
st.set_page_config(layout="wide", page_title='Further Evaluation', page_icon='🤔')




import plotly.express as px

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud, STOPWORDS


st.title('Misclassifications')
st.markdown("""---""")
st.write("###")
df = pd.read_excel('./Data/data-for-capstone.xlsx')
df.columns = df.columns.str.lower()
df = df.rename(columns={'spam?':'spam'})
X = df['text']
y = df['spam']
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y, random_state=42)


mnb_pipeline = Pipeline([
    ('cv', CountVectorizer(stop_words='english', min_df = 1, ngram_range = (1,1))),
    ('mnb', MultinomialNB(alpha=0.1))
])

mnb_pipeline.fit(X_train, y_train)

test_preds = mnb_pipeline.predict(X_test)
pd.set_option('display.max_colwidth', None)

misclassifications = pd.DataFrame(X_test)

misclassifications['pred class'] = test_preds
misclassifications['true class'] = y_test

mis = misclassifications[misclassifications['pred class'] != misclassifications['true class']]
st.write(mis, use_container_width='True')
