#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#TSB 11/27/17
#Use housing price dataset from kaggle (kings county, Washington) with tensorflow
#based on andrew geitgey's ML class on lynda.com
Created on Mon Nov 27 16:53:09 2017

@author: hsantanam
"""

import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load training data set from CSV file
df = pd.read_csv("/Users/hsantanam/Downloads/kc_house_data.csv")
# Pull out columns for X (data to train with) and Y (value to predict)
del df['id']
del df['date']
del df['zipcode']
del df['lat']
del df['long']

y = df['price'].as_matrix()
del df['price']

X = df.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#the lines below are for arrays with one vector/shape.  Without this, system threw an error and prompted me to include this.
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# All data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well. Create scalers for the inputs and outputs.
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_train)
Y_scaled_training = Y_scaler.fit_transform(y_train)

# It's very important that the training and test data are scaled with the same scaler.
X_scaled_testing = X_scaler.transform(X_test)
Y_scaled_testing = Y_scaler.transform(y_test)

# Define model parameters
learning_rate = 0.001
training_epochs = 100

# Define how many inputs and outputs are in our neural network
number_of_inputs = 15
number_of_outputs = 1

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# Section One: Define the layers of the neural network itself

# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))
    
# Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable("weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable("weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable("weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights) + biases

# Section Two: Define the cost function of the neural network that will measure prediction accuracy during training

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# Section Three: Define the optimizer function that will be run to optimize the neural network

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialize a session so that we can run TensorFlow operations
with tf.Session() as session:

    # Run the global variable initializer to initialize all variables and layers of the neural network
    session.run(tf.global_variables_initializer())

    # Run the optimizer over and over to train the network.
    # One epoch is one full run through the training data set.
    for epoch in range(training_epochs):

        # Feed in the training data and do one step of neural network training
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

        # Every 5 training steps, log our progress
        if epoch % 5 == 0:
            training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y:Y_scaled_training})
            testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y:Y_scaled_testing})

            print(epoch, training_cost, testing_cost)

    # Training is now complete!
    print("Training is complete!")

    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))
