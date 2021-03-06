# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:56:08 2015

@author: dpryce
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.cross_decomposition import PLSRegression
import csv

'''DATA CHECKS'''
#print train_df.dtypes
##identified date, city, city group and type that need munging/cleaning
## also all values need to set to float before modelling
#print train_df.info()
## identified there are no null values but must check if zero values mean anything

def munge_rest(df):    
    '''DROP ID'''
    df = df.drop(['Id'],axis=1)
    
    '''MUNGE DATE FEATURES'''
    #Create new column for date and copy from open date into pd datetime
    df['Date'] = df['Open Date'].apply(pd.to_datetime)
    
    ##Use the below to set the index as date instead of dataframe id
    df.set_index('Date', inplace=True)
    
    ##Extract variables from date
    df['OpenDay'] = df.index.day
    df['OpenMonth'] = df.index.month
    df['OpenYear'] = df.index.year
    df = df.reset_index(drop=True)
    
    #Drop the open date column
    df = df.drop(['Open Date'],axis=1)
    
    '''MUNGE CITY FEATURE'''
    #creating a dictionary for the cities each unique city has a value
    unique_cities = pd.unique(df.City.ravel())
    values = np.arange(len(unique_cities))
    city_dict = dict(zip(unique_cities,values))
    #İstanbul: 0, Ankara: 1, Diyarbakır:2, Tokat: 3, Gaziantep: 4, 
    #Afyonkarahisar 5, Edirne: 6, Kocaeli: 7, Bursa: 8, İzmir: 9,
    #Sakarya: 10, Elazığ: 11, Kayseri: 12, Eskişehir: 13, Şanlıurfa: 14,
    #Samsun: 15, Adana: 16, Antalya: 17, Kastamonu: 18, Uşak: 19, Muğla: 20
    #Kırklareli: 21, Konya: 22, Karabük: 23, Tekirdağ: 24, Denizli: 25
    #Balıkesir: 26, Aydın: 27, Amasya: 28, Kütahya: 29, Bolu: 30, Trabzon: 31
    #Isparta: 32, Osmaniye: 33
    
    #NOTE: "Tanımsız" means undefined in Turkish - language used here
    
    #Adding new column
    df['CityID']=df['City']
    
    #Replacing all city strings with dictionary values
    df=df.replace({"CityID": city_dict})
    
    #Drop the city string column
    df = df.drop(['City'],axis=1)
    
    
    '''MUNGE CITY GROUP FEATURE'''
    # City Group: Type of the city. Big cities, or Other. 
    #Create a dictionary for this
    group_dict = {"Big Cities": 0, "Other": 1}
    #create a new column to work on
    df['GroupID']=df['City Group']
    #replace values from dictionary
    df = df.replace({'GroupID':group_dict})
    ##check the two columns agree
    #print df[['City Group','GroupID']]
    #Drop the city group string column
    df = df.drop(['City Group'],axis=1)
    
    '''MUNGE TYPE FEATURE'''
    #Type: Type of the restaurant. FC: Food Court, IL: Inline, DT: Drive Thru, MB: Mobile
    #Create dictionary for this
    type_dict = {"FC": 0, "IL": 1, "DT": 2, "MB": 3}
    #copy column to work on
    df['TypeID']=df['Type']
    #replace values from dictionary
    df = df.replace({'TypeID':type_dict})
    ##check the two columns agree
    #print df[['Type','TypeID']]
    #Drop the Type string column
    df = df.drop(['Type'],axis=1)
    
    '''FINAL DATA CHECKS'''
    #print df.dtypes
    ##all values are int or float
    #print df.info()
    # same as before need to invesigate if zero means anything
    
    #print train_values
    
    #drop dataframe in a numpy array
    data = df.values
    return data

#change below for your local file directory 
os.chdir('C:\Users\dpryce\Documents\DataAnalytics\Kaggle - restaurant')

'''DATA INPUT'''
#input training data into initial pandas dataframe
train_df = pd.read_csv('train.csv',sep=",")
test_df = pd.read_csv('test.csv',sep=",")   
    
train_data = munge_rest(train_df)
test_data = munge_rest(test_df)    
    
#Delete revenue column from train data
x = np.delete(train_data,37,1)
revenue = train_data[:,37]

'''TRAINING'''
'''Set up Partial Least Squares Regression'''
pls = PLSRegression(n_components=25)

'''Fit model'''
pls = pls.fit(x, revenue)


'''TESTING'''
'''Predict and score'''
#Apply Partial least squares regressor to test data
prediction = pls.predict(test_data)


''' PRINT TO FILE'''
predictions_file = open("restaurants_pls.csv", "wb")
p = csv.writer(predictions_file)
p.writerow(["Id", "Prediction"])

for i in range(0,len(prediction)):
    p.writerow([i, prediction[i,0]])
    
predictions_file.close()