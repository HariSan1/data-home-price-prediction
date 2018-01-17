#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# TSB  simple classifier prediction 
Created on Thu Nov 30 22:36:49 2017

@author: hsantanam
"""

import pandas as pd
from sklearn import tree
training_data_df = pd.read_csv("/Users/hsantanam/Downloads/Ex_Files_TensorFlow/Exercise Files/04/sales_data_training.csv", dtype=float)

# Pull out columns for X (data to train with) and Y (value to predict)
X_training = training_data_df.drop('total_earnings', axis=1).values
Y_training = training_data_df[['total_earnings']].values

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_training, Y_training)
print(clf.predict([[4,1,0,1,0,1,0,1, 39.99]]))