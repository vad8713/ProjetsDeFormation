# -*- coding: utf-8 -*-
import numpy as np
import streamlit as st

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from joblib import load

import nltk
nltk.download('punkt')
nltk.download('omw-1.4')
#nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
import re

import tensorflow as tf
import tensorflow_hub as hub

def tokenizer_fct(sentence) :
    # print(sentence)
    #sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    sentence_clean = sentence.replace('-', ' ').replace('&', ' ').replace('/', ' ')
    word_tokens = word_tokenize(sentence_clean)
   
    return word_tokens

def retokenizer_fct(tokens) :
    mwtokenizer = nltk.MWETokenizer(separator='')
    mwtokenizer.add_mwe(('c', '#'))
    word_tokens = mwtokenizer.tokenize(tokens)
    
    return word_tokens

# Stop words
#from nltk.corpus import stopwords
#stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')',';','!','<','>']

# lower case et alpha
def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words if (not w.startswith("@")) 
    #                                   and (not w.startswith("#"))
                                       and (not w.startswith("http"))]
    return lw

# remove code and tags in pattern : <code> code </code>)
def removeCodeMarkup(sentence):
    codeMarkupRegEx = r'<code>(.*?)</code>'
    #re.compile(codeMarkupRegEx,flag=re.DOTALL)
    cleanText = re.sub(codeMarkupRegEx,'',sentence,flags=re.DOTALL)
    return cleanText

# remove html tags
def removeHTML(sentence):
    htmlMarkupRegEx = '<.*?>'
    cleanText = re.sub(htmlMarkupRegEx,'',sentence)
    return cleanText

# Fonction de préparation du texte pour le Deep learning (USE et BERT)
def transform_dl_fct(desc_text) :
    sentence = removeCodeMarkup(desc_text)
    sentence = removeHTML(sentence)
    word_tokens = tokenizer_fct(sentence)
#    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(word_tokens)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

def process_log_regression_model(x, y, isfit=False, isScore=True):

    if isfit == True:
        print("Processing Logistic Regression with fit")
        xscaled = scaler.fit_transform(x)
        xreduct = pca.fit_transform(xscaled)
        lr.fit(xreduct, y)
    else:
        print("Processing Logistic Regression")
        xscaled = scaler.transform(x)
        xreduct = pca.transform(xscaled)
    
    ypred = lr.predict(xreduct)
    if isScore == True:
        score = np.round(lr.score(xreduct,y),2)
    else:
        score = 0
        
    return xreduct, ypred, score

@st.cache_resource(ttl=24*3600)
def long_running_function():
    # load models
    le = load(labels_filename)
    scaler = load(scaler_filename)
    pca = load(pca_filename)
    lr = load(logreg_filename)
    
    # load USE model
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    #embed = hub.load("local-universal-sentence-encoder_4")
    
    return le,scaler,pca,lr,embed

# add models
labels_filename = 'labelsEncoder.joblib'
scaler_filename = 'scale.joblib'
pca_filename = 'pca.joblib'
logreg_filename = 'logisticRegression.joblib'

le,scaler,pca,lr,embed = long_running_function()
#st.write("labels classes are : ",le.classes_)

st.write("""
# Tag proposal application
""")

entryText = st.text_input('Enter text below :', value = "")
if entryText != "":
    formatedText = transform_dl_fct(entryText)
    #st.write("**Formated text is** :\n", formatedText)
    feat = embed([formatedText])
    x, ypred, score = process_log_regression_model(feat,0,isScore=False)
    label = le.inverse_transform(ypred)[0]
else:
    label = ""
st.write("**Suggested Labels is** :\n", label)
