# importing the module
import json
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
import re
import csv

# function to remove special characters
def preprocess_sentence(text,flg_stemm=False,flg_lemm=False):
    # Removing punctuations
    pat = r'[^a-zA-z0-9\s]' 
    #return re.sub(pat, '', text)
    text = re.sub(pat, '', text)
    text = str(text).lower().strip()
    lst_text = text.split()
    lst_stopwords = set(stopwords.words('english')) 
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
    text = " ".join(lst_text)
    return text;


def spacy_sentence_embeddings(sentence,nlp): #Let's get some static embeddings
    C=[]
    sentence = preprocess_sentence(sentence)
    if len(sentence)==0:
        sentence = "empty"
    doc = nlp(sentence)
    for i in range(len(doc)):
        C.append(doc.vector) #Word embedding
    word_embed_mean = [sum(x)/len(doc) for x in zip(*C)]
    return word_embed_mean   #Returns mean of all word embeddings in the string



def get_feature_embedding(fs,nlp,raw_data):
    result_word_embeddings = []
    for i in range(len(raw_data)):
        s1="empty" #Taking this embedding if any of column value is empty
        s2 = ""
        temp = []
        if raw_data['seniority_level'][i]: #Ignoring row if seniority_level is empty
            for j in range(len(fs)):
                if len(raw_data[fs[j][0]][i])!=0:
                    s2 = raw_data[fs[j][0]][i][0][fs[j][1]] #Taking only first row
                    if type(s2) is list:
                        s2 = " ".join(map(str, s2));
                if s2=="" or not s2 or len(s2)==0 or s2==" ":
                    temp = temp + spacy_sentence_embeddings(s1,nlp)
                else:
                    temp = temp + spacy_sentence_embeddings(s2,nlp)
            result_word_embeddings.append(temp)
        print(i)
    return result_word_embeddings

def dump_embeddings(f1,field_subfield_map,dump_file,nlp):
    raw_data = pd.read_json(f1,
                        lines=True,
                        orient='columns')
    embeddings_list = get_feature_embedding(field_subfield_map,nlp,raw_data)
    with open(dump_file,"w") as f:
        wr = csv.writer(f)
        wr.writerows(embeddings_list)

#field_subfield_map = [['experience','title'],['experience','skills'],['education','school'],['education','degree']]
#for all fields 75% accuracy
#for only title 76% accuracy
#for title+skills
field_subfield_map = [['experience','title'],['experience','skills']]
nlp = spacy.load('en_core_web_md')
dump_embeddings("seniority.train",field_subfield_map,"train_embeddings.csv",nlp)
dump_embeddings("seniority.test",field_subfield_map,"test_embeddings.csv",nlp)


