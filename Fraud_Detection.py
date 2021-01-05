#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 20:34:49 2021

@author: soumya
"""
# Python >= 3.8.5
# numpy >= 1.19.2
# pandas >= 1.1.5
# matplotlib >= 3.3.2
# seaborn >= 0.11.1
# scipy >= 1.5.2
# sklearn >= 0.23.2

############################### Import Modules ###############################

import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print(' ')
print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__))
print('Sklearn: {}'.format(sklearn.__version__))

import pandas as pd # to load and manipulate data
import numpy as np # to calculate the mean and standard deviation
import matplotlib.pyplot as plt # to draw graphs
import seaborn as sns # for better plotting

 ################################ Import Data #################################

# now pandas reads the data into a dataframe (df). 
# The function in pandas to read csv files is read_csv
# the csv file needs to be in the same directory as the script
data = pd.read_csv('creditcard.csv')

# check the name of the columns
print(' ')
print('Names of the columns:')
print(data.columns)

# shape of the data
print(' ')
print('Size of the original data:', data.shape)
# there are around 300,000 transactions with 30 attributes such as
# Time -- time interval between two consecutive transaction
# Amount -- amount of money in the transaction
# Class = 0 -- normal transaction
#       = 1 -- fradulant transaction

# to look at the statistics of each attribute
print(' ')
print(data.describe())

############################ Investigating Dataset ###########################

# this is a large dataset. instead of using the entire dataset, I am going to 
# use 10% of the data, and to get the same 1% of the data, we defined a random state
data = data.sample(frac = 0.99, random_state = 1)
print(' ')
print('Size of the trimmed data:', data.shape)

# we can further explore the data by looking at the histogram for each attribute
data.hist(figsize = (20, 20))
plt.show()

# determine the number of fraud cases and normal cases in the data
Valid = data[data['Class'] == 0]
Fraud = data[data['Class'] == 1]
outlier_frac = len(Fraud)/float(len(Valid)) # fraction of fraud cases
print(' ')
print('Number of valid cases:', len(Valid))
print('Number of fraud cases:', len(Fraud))
print('Fraction of fraud cases (outlier):', outlier_frac)

# build correlation matrix to check if there is any correlation between variables
cormap = data.corr()
# plotting correlation between parameters: correlation between Class and other parameters
fig = plt.figure(figsize = (12, 9))
sns.heatmap(cormap, vmax = 0.8, square = True)
plt.show()

# correctly format data for modeling
# get all the columns from the data frams
columns = data.columns.tolist()

# filter the columns to remove data we do not want. in this case we are interested
# in the 'Class' column
columns = [c for c in columns if c not in ['Class']]

# We are using unsupervised learning, so we cannot feed the labels to the model
# ahead of time during training
# we seperate the dependent and independent variable
target = 'Class' # variable we will be predicting on

X = data[columns]
Y = data[target]

# we can print the shape of the variables
print(' ')
print('Shape of the independent variable (X): ',X.shape)
print('Shape of the dependent variable (Y): ',Y.shape)

############################# Defining Classifiers ###########################
# isolated forest algorithms and local outlier factor algorithm to detect 
# anamoly in the dataset

# to determine how successful we are in detecting outlier
from sklearn.metrics import classification_report, accuracy_score

# importing IsolationForest and LocalOutlierFactor algorithm for modeling
# its a random forest algorithm that detect and score outliers
from sklearn.ensemble import IsolationForest
# unsuperviser outlier detection method, very similar to k-nearest neighbor method
from sklearn.neighbors import LocalOutlierFactor

 # we can also use support vector machine (svm) to detect outliers, but svm usually
# takes longer to train and more complicated to build

state = 1 # defining a random state
# include the outlier detection algorithms in a dictionary of classifiers
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination = outlier_frac, # initial guess of outlier fraction
                                        random_state = state),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, # default is 20, usually higher the contamination, higher the n_neighbor
                                               contamination = outlier_frac)
    }

############################## Fitting the Model #############################
n_outliers = len(Fraud)

print(' ')
print('------------------ Fitting Result ------------------')
# we will do for loop over two classifiers defined above
for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fitting data tagging outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X) # fit the X data and give labels
        scores_pred = clf.negative_outlier_factor_
        
    else: # for Isolation Forest
        clf.fit(X)
        score_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    # the y_pred will give us -1 for outliers and 1 for valid transactions
    # we need 1 for outliers (fraud) and 0 for valid transactions to be 
    # consistent wit the dataset
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    # estimating error
    n_errors = (y_pred != Y).sum()
    
    # printing results
    # 1. print the name of the classification and the number of missclassifications (errors)
    print('{}: {}'.format(clf_name, n_errors))
    # 2. print the accuracy score
    print(accuracy_score(Y, y_pred))
    # 3. print the classification matrix
    print(classification_report(Y, y_pred))
    
# the result should look like this for 90% of the data
"""
------------------ Fitting Result ------------------
Isolation Forest: 665
0.9976415010693044
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    281469
           1       0.32      0.32      0.32       490

    accuracy                           1.00    281959
   macro avg       0.66      0.66      0.66    281959
weighted avg       1.00      1.00      1.00    281959

Local Outlier Factor: 931
0.9966981014970262
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    281469
           1       0.05      0.05      0.05       490

    accuracy                           1.00    281959
   macro avg       0.52      0.52      0.52    281959
weighted avg       1.00      1.00      1.00    281959
"""
