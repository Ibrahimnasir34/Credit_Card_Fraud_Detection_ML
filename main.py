#importing the dependices
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from  sklearn.metrics import  accuracy_score
import pickle
#loading data set pandas dataframe

credit_cardData=pd.read_csv('creditcard.csv')

#printing 5 rows
print(credit_cardData.head())

# the admin has converted ass using pricipal component analysis method converted all the features to numerical value

#last five
print(credit_cardData.tail())

#dataset information
print(credit_cardData.info())

#checking number of missing values in each column

print(credit_cardData.isnull().sum())

#class 0 fraud class 1 legits

#distribution of legit transcation and fraudulent transactions

print(credit_cardData['Class'].value_counts())

#data is very imbalanced very less data pounts for fraud
# data processing
#separate the data for analysis
legit =credit_cardData[credit_cardData.Class==0]
fraud =credit_cardData[credit_cardData.Class==1]

print(legit.shape)
print(fraud.shape)

#sattsiscal measures of the data

legit.Amount.describe()

fraud.Amount.describe()
#important for diferrences

#compare the values for both transaction
credit_cardData.groupby('Class').mean()
#gives for the mean of every columns by groupong in class 0 and 1 the differnece is impotant

#under-sampling
#build a sample dataset containing similar distribution of npormal and fraud transactions

#fraud 492
#legit 492
legit_sample=legit.sample(n=492)

#concatenating 2 dataframes


#axis =0 means concatenate row wise
#axis =1 means concatenate column wise
new_dataset=pd.concat([legit_sample,fraud],axis=0)
print(new_dataset.head())
print(new_dataset.tail())
#to count values of classs
new_dataset['Class'].value_counts()

#mean values
print(new_dataset.groupby('Class').mean())

#splitiing the data into featues and targets
#drop class by cloumns wise
X=new_dataset.drop(columns='Class',axis=1)

Y=new_dataset['Class']

print(X)
print(Y)

#startify start the data at y distribution of 0,1 can be differnet in train and test if not used  randowm starte how you want to split data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

#print(X.shape(),x_train.shape(),x_test.shape())

#model training
#logistics regreasssion for binary classfication problems

model=LogisticRegression()

#training the logistic regression model with training data

model.fit(x_train,y_train)


#evalutaion
#accuracy score
x_train_prediction=model.predict(x_train)
trainingdataaccir=accuracy_score(x_train_prediction,y_train)

print(":accuracy score on traing data ",trainingdataaccir)
#test
x_test_prediction=model.predict(x_test)
testdataaccir=accuracy_score(x_test_prediction,y_test)
print(":accuracy score on test data ",trainingdataaccir)


with open('logistic_regression_model_For_credit_card_fraud.pkl', 'wb') as file:
    pickle.dump(model, file)