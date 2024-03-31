# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 23:11:09 2024

@author: suraj
"""

#navie bayes on the nb_car_ad dataset
"""
In this data set there are 400 labels/row and 5 columns are present
User ID: This column is likely a unique identifier for each user.
 It might contain alphanumeric codes or numbers. 
The data type is likely to be a string (or possibly integer, depending on the format).

Gender: This column represents the gender of each user. 
It might contain categorical values like "Male" or "Female."
 The data type is likely to be a string or categorical.

Age: This column represents the age of each user. 
It is likely to contain numerical values. 
The data type is likely to be an integer.

Estimated Salary: This column provides an estimate of the salary or 
income of each user. It is likely to contain numerical values. 
The data type is likely to be a floating-point number.

Purchased: This column is binary and indicates whether a user made a
 purchase or not. It might have values like 0 and 1, where 0 represents 
 "Not Purchased" and 1 represents "Purchased." The data type is likely 
 to be an integer or boolean.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

arr=[]
car=pd.read_csv("C:/Data Science/14_a_ML_Assigement/1_Naive_bayes_assegemt/dataset/NB_Car_Ad.csv")
car.columns
##################################
car.shape    # (400, 5)
##################################
car.head(10)
##################################
a=car.isnull()
a.sum()
#it does not having any null value
################################
car.dtypes
'''Out[90]: 
User ID             int64
Gender             object
Age                 int64
EstimatedSalary     int64
Purchased           int64
dtype: object'''

###############################
car.describe()
##################
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

cleaning_text("male")
cleaning_text("female")

purchased= {0: 'Not Purchased', 1: 'Purchased'}
# Replace integers with strings using the mapping in the 'Purchased' column
car['Purchased'] = car['Purchased'].replace(purchased)
car                 # converting the number to string

car.Gender=car.Gender.apply(cleaning_text)
car=car.loc[car.Gender !=" ",:]
car
#######################
from sklearn.model_selection import train_test_split
car_train,car_test=train_test_split(car, test_size=0.2)
#for making the data modeling and testing the data split into the chunk
car_train
car_test
###############
#creating matrix of token counts for entire gender document

def split_into_words(i):
    return[word for word in i.split(" ")]

car_pot= CountVectorizer(analyzer=split_into_words).fit(car.Gender)
all_car_matrix=car_pot.transform(car.Gender)
#for traning the model
train_car_matrix=car_pot.transform(car_train.Gender)
#for testing the meassage
test_car_matrix=car_pot.transform(car_test.Gender)


#learning term weightaging and normaling on entire gender

tfidf_transformer=TfidfTransformer().fit(all_car_matrix)

#preparing TFIDF for train gender
train_tfidf=tfidf_transformer.transform(train_car_matrix)
#preparing TFIDF tor tsting the gender
test_tfidf=tfidf_transformer.transform(test_car_matrix)
test_tfidf.shape

##################
#Now let us apply to the navie bayes 
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb=MB()
# we are traning the data we can use the Salary 
classifier_mb.fit(train_tfidf,car_train.Purchased) #Out[47]: MultinomialNB() if it execute proper
#evaluation on t data

# for the salary column and eductaion column we can predicte it
test_pred_m=classifier_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==car_test.Purchased)
accuracy_test_m                 #Out[72]: 1.0
# it having the 92% accuracy when we apply the navie bayes in the two column
##########################
# now we can check for the another column it hoursperweek column and education column
classifier_mb.fit(train_tfidf,car_train.Age) #Out[47]: MultinomialNB() if it execute proper
#evaluation on t data

# for the salary column and eductaion column we can predicte it
test_pred_m=classifier_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==car_test.Age)
accuracy_test_m      # 0.0875
# it having the 8% accuracy when we apply the navie bayes in the two column
#####################################
''' from the above model we can coclude that the people is more age that
people is less purchased'''