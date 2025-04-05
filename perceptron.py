#-------------------------------------------------------------------------
# AUTHOR: Tam Tran
# FILENAME: perceptron.py
# SPECIFICATION: Compare Perceptron and MLPClassifier on the optdigits dataset
# FOR: CS 4210- Assignment #3
# TIME SPENT: ~ 45 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]
# Load training data
df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

# Load testing data
df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

#Initialize the highest accuracy for each algorithm
highest_perceptron_accuracy = 0
highest_mlp_accuracy = 0

# Iterate all combinations of learning rates and shuffle
for rate in n: #iterates over n

    for shuffle in r: #iterates over r

        #iterates over both algorithms
        for model in ['Perceptron', 'MLPClassifier']:
            #Create a Neural Network classifier
            if model == 'Perceptron':
                clf = Perceptron(eta0=rate, shuffle=shuffle, max_iter=1000) #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            else:
                clf = MLPClassifier(activation='logistic', #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
                                    learning_rate_init=rate, 
                                    hidden_layer_sizes=(25,), # hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
                                    shuffle=shuffle, 
                                    max_iter=1000) 
            
            clf.fit(X_training, y_training) 
        
            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            correct_predictions = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test): #iterates over the algorithms
                #make a prediction for the test sample
                prediction = clf.predict([x_testSample]) 
                #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
                if prediction[0] == y_testSample: 
                    correct_predictions += 1 #increment the number of correct predictions

            accuracy = correct_predictions / len(X_test) #calculate the accuracy

            # Update the highest accuracy for each algorithm
            if model == 'Perceptron' and accuracy > highest_perceptron_accuracy:
                highest_perceptron_accuracy = accuracy
                print(f"Highest Perceptron accuracy so far: {accuracy:.3f}, Parameters: learning rate={rate}, shuffle={shuffle}")
            elif model == 'MLPClassifier' and accuracy > highest_mlp_accuracy:                    
                highest_mlp_accuracy = accuracy
                print(f"Highest MLP accuracy so far: {accuracy:.3f}, Parameters: learning rate={rate}, shuffle={shuffle}")