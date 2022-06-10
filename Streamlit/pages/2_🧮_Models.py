import streamlit as st
st.set_page_config(layout="wide", page_title='Models', page_icon="🧮")





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
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


df = pd.read_csv('data-for-capstone.csv')
df.columns = df.columns.str.lower()
df = df.rename(columns={'spam?':'spam'})
X = df['text']
y = df['spam']
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y, random_state=42)


###############################


st.title('Models')
st.markdown("""---""")
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)


###############################


#multinomial naive bayes
st.subheader('Multinomial Naive Bayes ♛')

mnb_pipeline = Pipeline([
    ('cv', CountVectorizer(stop_words='english', min_df = 1, ngram_range = (1,1))),
    ('mnb', MultinomialNB(alpha=0.1))
])

mnb_pipeline.fit(X_train, y_train)

col1, col2, = st.columns(2)

with col1:
    st.text(('Classification Report:\n ' + classification_report(y_test, mnb_pipeline.predict(X_test))))
with col2:
    mnb_cm = confusion_matrix(y_test, mnb_pipeline.predict(X_test))
    st.text('Confusion matrix:')
    st.write(mnb_cm)

with st.expander("See code"):
    mnb_code = '''pipeline = Pipeline([
    ('cv', CountVectorizer(stop_words='english')),
    ('mnb', MultinomialNB())
])

parameters = {
    'cv__min_df': (1,2,3),
    'cv__ngram_range': ((1, 1), (1, 2), (2,2)),
    'mnb__alpha': [0.01, 0.05, 0.1]
}
    
gs_mnb = GridSearchCV(pipeline, param_grid = parameters, n_jobs=-1)
gs_mnb.fit(X_train, y_train);'''

    st.code(mnb_code, language='python')

st.write("#")

with open("saved_model.pkl", 'wb') as file:
    pickle.dump(mnb_pipeline, file)

st.markdown("""---""")






###############################






#random forest classifier
st.subheader('Random Forest Classifier')

rfc_pipeline = Pipeline([
    ('cv', CountVectorizer(stop_words='english', min_df = 3, ngram_range = (1,2))),
    ('rfc', RandomForestClassifier(random_state=42, max_depth = None, min_samples_leaf = 1, n_estimators = 300))
])

rfc_pipeline.fit(X_train, y_train)

col1, col2, = st.columns(2)
with col1:
    st.text(('Classification Report:\n ' + classification_report(y_test, rfc_pipeline.predict(X_test))))
with col2:
    rfc_cm = confusion_matrix(y_test, rfc_pipeline.predict(X_test))
    st.text('Confusion matrix:')
    st.write(rfc_cm)

with st.expander("See code"):
    rfc_code = '''
    pipeline = Pipeline([
    ('cv', CountVectorizer(stop_words='english')),
    ('rfc', RandomForestClassifier(random_state=42))
])

parameters = {
    'cv__min_df': (1,3),
    'cv__ngram_range': ((1, 1), (1, 2)),
    'rfc__n_estimators': (300, 500),
    'rfc__max_depth': (None, 3, 5),
    'rfc__min_samples_leaf': (1, 3)
}
    
gs_rfc = GridSearchCV(pipeline, param_grid = parameters, n_jobs=-1)
gs_rfc.fit(X_train, y_train);'''

    st.code(rfc_code, language='python')

st.write("#")


st.markdown("""---""")








###############################








#Logistic Regression
st.subheader('Logistic Regression')

lr_pipeline = Pipeline([
    ('cv', CountVectorizer(stop_words='english', min_df = 2, ngram_range = (1,1))),
    ('lr', LogisticRegression(max_iter = 10_000, C=1.0))
])
    
lr_pipeline.fit(X_train, y_train)

col1, col2, = st.columns(2)
with col1:
    st.text(('Classification Report:\n ' + classification_report(y_test, lr_pipeline.predict(X_test))))
with col2:
    lr_cm = confusion_matrix(y_test, lr_pipeline.predict(X_test))
    st.text('Confusion matrix:')
    st.write(lr_cm)


with st.expander("See code"):
    lr_code = '''
    lr = LogisticRegression(max_iter=10_000)
    lr.fit(X_train_cv, y_train)
    pipeline = Pipeline([
        ('cv', CountVectorizer(stop_words='english')),
        ('lr', LogisticRegression(max_iter = 10_000))
    ])
    parameters = {
        'cv__min_df': (1,2,3),
        'cv__ngram_range': ((1, 1), (1, 2)),
        'lr__C': [0.25, 0.5, 0.75, 1.0]}
        
    gs_lr = GridSearchCV(pipeline, param_grid = parameters)
    gs_lr.fit(X_train, y_train)'''

    st.code(lr_code, language='python')