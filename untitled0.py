import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sb 
import matplotlib.pyplot as plt 
import streamlit.components.v1 as components
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
plt.style.use('dark_background') ; 
st.set_option('deprecation.showPyplotGlobalUse', False)
components.html(
    """
    <div class="intro" style="border: 2px solid black; border-radius: 25px; background-color: cornflowerblue;font-family:Trebuchet MS,Garamond ;font-size: 18px;text-align: center; ">
        <h1 style="margin: 0.35;"><strong>Job-Detect</strong></h1>
        <h2 style="margin: 0.35;">An app that can determine real and fake job postings.</h2>
    </div>
    <br>
  
    """,
    height=350,
)

model = pickle.load(open(r'D:\VII th Sem\model_final', 'rb')) 
tfidf = pickle.load(open(r'D:\VII th Sem\tfidf', 'rb'))
with st.form('my_form'):
    sentence = st.text_input("Enter job description...")
    x=st.form_submit_button(label="Submit")

if(x):
    st.write(type(sentence))
    #tfidf= TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
    
    inp= tfidf.transform(pd.Series(sentence))
    #st.write(desc,'\n',inpu)
    ans = model.predict(inp)
    
    if(ans==1): st.success("Job is Not Fraudulent. ")
    elif(ans==0): st.error("Job is Fraudulent. ")
    else: st.write('')