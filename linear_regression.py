# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:20:29 2022

@author: Michu
"""

import numpy as np 
import pandas as pd
import datetime
from tqdm import tqdm

#Set testing mode with a reduced amount of data
TESTING = False


# splits data into training and testing
def split_data(X, y):
    np.random.permutation(X)
    train_size = int(0.7 * len(X))
    xtrain_indices = X[:train_size]
    xtest_indices = X[train_size:]
    ytrain_indices = y[:train_size]
    ytest_indices = y[train_size:]

    # split the actual data

    x_train, x_test = xtrain_indices.iloc[:,0:4], xtest_indices.iloc[:,0:4]
    y_train, y_test = ytrain_indices.iloc[:], ytest_indices.iloc[:]
    
    return x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy()


# Formats dates as seconds to handle time as a double
def get_seconds(complete_date):
    total_data["Date"] = total_data["Date"] + "/" + total_data["Time"].str.replace(":", "/")
    del total_data["Time"]

    complete_date = total_data["Date"]
    dates_formatted = complete_date.str.split("/")
    dates_formatted = dates_formatted.transform(lambda x: [int(x[0]), int(x[1]), int(x[2])\
                                                  , int(x[3]), int(x[4]) , int(x[5])])
    dates_formatted = dates_formatted.transform(lambda x : datetime.timedelta(x[0], x[1], x[2],\
                                                                              x[3], x[4], x[5]).total_seconds())
    total_data["Date"] = dates_formatted


# y = X*B + e
# e = X*B -y
# Fits multiple linear regression model
def fit_regression(X, y):
    #y = X*b + e
    #X nx3
    #Y n*1
    X_trans = np.transpose(X)  #3xn
    X_squared = np.dot(X_trans, X)  #3x3
    X_squared_inv = np.linalg.inv(X_squared)  #3x3
    b = np.dot(np.dot(X_squared_inv, X_trans), y)   #3x1
    e = np.subtract(np.dot(X, b), y)  # variation in each point
    return b, e
    

accuracy = 1


# Execute algorithm to guess temperatures
def attempt(x_test, b, e, y_test):
    global accuracy
    guess = np.add(np.dot(x_test, b), np.abs(e).mean())  #y=b*x+e
    error = np.abs(np.subtract(y_test,guess))
    print("[-] Testing accuracy...")
    for i in tqdm(error, total=len(error)):
        accuracy /= accuracy + error[0] 
    accuracy = 1 - accuracy
    
    
# load the dataset
if TESTING:
    temperatures=pd.read_csv("testing.csv",delimiter=',', encoding='unicode_escape',\
                            usecols=[6]).fillna(' ')
    total_data = pd.read_csv("testing.csv",delimiter=',', encoding='unicode_escape',\
                            usecols=[2,3,4,5,7]).fillna(' ')
else:
    temperatures=pd.read_csv("dataset.csv",delimiter=',', encoding='unicode_escape',\
                            usecols=[6]).fillna(' ')
    total_data = pd.read_csv("dataset.csv",delimiter=',', encoding='unicode_escape',\
                            usecols=[2,3,4,5,7]).fillna(' ')


#---------- Testing -----------
print("Handling data...")
get_seconds(total_data)                      
X = total_data
x_train, y_train, x_test, y_test = split_data(X,temperatures)
b, e = fit_regression(x_train, y_train)
attempt(x_test, b, e, y_test)
print("\n[-] The accuracy was of {}%".format(accuracy[0] * 100))














