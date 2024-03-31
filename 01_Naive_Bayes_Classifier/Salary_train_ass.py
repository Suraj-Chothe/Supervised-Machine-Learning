# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:39:37 2024

@author: suraj
"""

#########Salary_train 
#train and test dataset are given separately
#business objective
"""
In this data set there are 30161 labels/row and 14 columns are present
1.Age: The age of the individual | Relevant | 
2.Workclass: The type of employment or work arrangement (e.g., private, self-employed, government).| Relevant | 
3.Education: The highest level of education achieved by the individual.| Relevant
4.Educationno: The numerical representation of the education level.|Irrelevant
5.Maritalstatus: The marital status of the individual (e.g., married, single, divorced).| Relevant
6.Occupation: The type of job or profession the individual is engaged in. | Relevant
7.Relationship: The person's relationship status (e.g., husband, wife, own-child, unmarried). | Relevant
8.Race: The racial background or ethnicity of the individual. | Irrelevant
9.Sex: The gender of the individual (male or female). | Relevant
10.Capitalgain: The capital gains of the individual. | Relevant
11.Capitalloss: The capital losses of the individual.| Relevant
12.Hoursperweek: The number of hours the individual works per week.| Relevant
13.Native: Native country or place of origin.| Relevant
14.Salary: The income level or salary of the individual.| IrRelevant(Likely the target variable) 

 """


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


salary=pd.read_csv("C:/Data Science/14_a_ML_Assigement/1_Naive_bayes_assegemt/dataset/SalaryData_Train.csv")
salary.columns
##################################
salary.shape    #(30161, 14)
##################################
salary.head(10)
##################################
a=salary.isnull()
a.sum()
#it does not having any null value
##################################
salary.dtypes
'''Out[77]: 
age               int64
workclass        object
education        object
educationno       int64
maritalstatus    object
occupation       object
relationship     object
race             object
sex              object
capitalgain       int64
capitalloss       int64
hoursperweek      int64
native           object
Salary           object
dtype: object'''
##############################
salary.describe()
###############################
#cleanning the data

import re
def cleaning_text(i):
    w=[]
    i=re.sub("[^A-Za-z0-9""]+"," ",i).lower()
    for word in i.split(" "):
        if len(word)>0:
            w.append(word)
    return(" ".join(w))


##############
# testing the above function with some test text
# fro the labeling we are taking the "education" column and taking the another 
#feature for taking to "eductaionno ,hoursperweek, salary' column 

cleaning_text("State-gov")
cleaning_text("13th")
salary.education=salary.education.apply(cleaning_text)
salary=salary.loc[salary.education !=" ",:]
#######################
from sklearn.model_selection import train_test_split
#import the testing file i.e Salarydata_test
salary_t=pd.read_csv("C:/Data Science/14_a_ML_Assigement/1_Naive_bayes_assegemt/dataset/SalaryData_Test.csv")
# we want to clean this test file also
salary_t.education=salary_t.education.apply(cleaning_text)
salary_t=salary_t.loc[salary_t.education !=" ",:]
salary_train,salary_t=train_test_split(salary)
#for making the data modeling and testing the data split into the chunk

###############
#creating matrix of token counts for entire education document

def split_into_words(i):
    return[word for word in i.split(" ")]

salary_pot= CountVectorizer(analyzer=split_into_words).fit(salary.education)
all_salary_matrix=salary_pot.transform(salary.education)
#for traning the model
train_salary_matrix=salary_pot.transform(salary_train.education)
#for testing the meassage
test_salary_matrix=salary_pot.transform(salary_t.education)


#learning term weightaging and normaling on entire education

tfidf_transformer=TfidfTransformer().fit(all_salary_matrix)

#preparing TFIDF for train education
train_tfidf=tfidf_transformer.transform(train_salary_matrix)
#preparing TFIDF tor tsting the education
test_tfidf=tfidf_transformer.transform(test_salary_matrix)
test_tfidf.shape

##################
#Now let us apply to the navie bayes 
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb=MB()
# we are traning the data we can use the Salary 
classifier_mb.fit(train_tfidf,salary_train.Salary) #Out[47]: MultinomialNB() if it execute proper
#evaluation on t data

# for the salary column and eductaion column we can predicte it
test_pred_m=classifier_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==salary_t.Salary)
accuracy_test_m                 #Out[72]: 0.7692307692307693
# it having the 63%(0.76/(0.76+0.46)) accuracy when we apply the navie bayes in the two column
##########################
# now we can check for the another column it hoursperweek column and education column
classifier_mb.fit(train_tfidf,salary_train.hoursperweek) #Out[47]: MultinomialNB() if it execute proper
#evaluation on t data

# for the salary column and eductaion column we can predicte it
test_pred_m=classifier_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==salary_t.hoursperweek)
accuracy_test_m      # 0.46578249336870026
# it having the 37%(0.46/(0.46+0.86)) accuracy when we apply the navie bayes in the two column
#####################################
''' from the above model we can coclude that the people is more salary that
people is more educated'''