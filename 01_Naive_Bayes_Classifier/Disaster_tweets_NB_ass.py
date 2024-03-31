# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 23:54:25 2024

@author: suraj
"""

#navie bayes on the Disaster_tweets_NB dataset
"""
'id':This column likely represents a unique identifier 
for each data entry. It might be a sequential number 
or some other form of identification.
'keyword':This column might contain specific keywords 
or phrases related to the data entries. In the context
 of NLP or text classification, these keywords could be 
 important for understanding the content of the text.
'location':This column may represent the location associated
 with each data entry. In the context of social media or 
 user-generated content, it could be the location mentioned 
 by the user.
'text':This column likely contains the main textual content 
of each entry. In NLP tasks, this is usually the primary focus,
 and models often analyze and classify text based on its content.
'target':This column appears to be the target variable or label 
for a classification task. It might indicate whether a particular
 entry belongs to a certain category or class. 
 For example, in sentiment analysis, it could represent whether 
 a text is positive, negative, or neutral.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


disa=pd.read_csv("C:/Data Science/14_a_ML_Assigement/1_Naive_bayes_assegemt/dataset/Disaster_tweets_NB.csv")
disa.columns
##################################
disa.shape    # (400, 5)
##################################
disa.head(10)
##################################
#describe the dataset
disa.describe()
#################################
#checking the column heading
disa.columns
'''Index(['id', 'keyword', 'location', 'text', 'target'], dtype='object')'''
################################
a=disa.isnull()
a.sum()
#it have the null values 
#################################
#we are use the the text and target column for the navie bayse algo so we drop the other column
disa=disa.drop(columns=['id','keyword','location'])
##################now the unneccasry data point are remove from the dataset
#the new dataset are as follows so we use the navie bayes algo 
disa
# checking any null point is present in the dataset 
b=disa.isnull()
b.sum()
##############################
#cleanning the data

import re
def cleaning_text(i):
    w=[]
    i=re.sub("[^A-Za-z0-9""]+"," ",i).lower()
    # Remove consecutive repeated letters and numbers
    i= re.sub(r'([a-zA-Z0-9])\1+', r'\1',i)
    for word in i.split(" "):
        if len(word)>2:
            w.append(word)
    return(" ".join(w))

################################
cleaning_text("phdsquares mufc they built much hype around acquisitions doubt they will ablaze this season")
disa.text=disa.text.apply(cleaning_text)
disa
w=disa.text.isnull()
w.sum()
disa=disa.loc[disa.text !=" ",:]
from sklearn.model_selection import train_test_split
disa_train,disa_test=train_test_split(disa,test_size=0.2)#for making the data modeling and testing the data split into the chunk
##############
#creating matrix of token counts for entire text document

def split_into_words(i):
    return[word for word in i.split(" ")]

disa_bow= CountVectorizer(analyzer=split_into_words).fit(disa.text)
all_disa_matrix=disa_bow.transform(disa.text)
#for traning the model
train_disa_matrix=disa_bow.transform(disa_train.text)
#for testing the meassage
test_disa_matrix=disa_bow.transform(disa_test.text)


#learning term weightaging and normaling on entire emails

tfidf_transformer=TfidfTransformer().fit(all_disa_matrix)

#preparing TFIDF for train mails
train_tfidf=tfidf_transformer.transform(train_disa_matrix)
#preparing TFIDF tor tsting the emails
test_tfidf=tfidf_transformer.transform(test_disa_matrix)
test_tfidf.shape

##################
#Now let us apply to the navie bayes 
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb=MB()
classifier_mb.fit(train_tfidf,disa_train.target) #Out[47]: MultinomialNB() if it execute proper
#evaluation on test data


test_pred_m=classifier_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==disa_test.target)
accuracy_test_m                                         #Out[29]: 0.8030203545633617


#############################
