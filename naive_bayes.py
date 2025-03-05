#-------------------------------------------------------------------------
# AUTHOR: Tam Tran
# FILENAME: naive_bayes.py
# SPECIFICATION: reads training and test data for weather conditions
#                to predict whether to play tennis based on Naive Bayes classification
# FOR: CS 4210- Assignment #2 - Problem 5e
# TIME SPENT: 45 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
def load_data(filename):
    """Loads data from a CSV file, skipping the header."""
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        return [row for row in csv_reader]
    
#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
def encode_features(data):
    """Transforms categorical features into numerical values and returns them as a list of lists."""
    outlook_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
    temperature_map = {'Hot': 1, 'Mild': 2, 'Cool': 3}
    humidity_map = {'High': 1, 'Normal': 2}
    wind_map = {'Strong': 1, 'Weak': 2}
    
    X = []
    for row in data:
        outlook = outlook_map[row[1]]
        temperature = temperature_map[row[2]]
        humidity = humidity_map[row[3]]
        wind = wind_map[row[4]]
        X.append([outlook, temperature, humidity, wind])
    return X

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
def encode_labels(data):
    """Transforms class labels into numbers."""
    play_map = {'Yes': 1, 'No': 2}
    
    Y = []
    for row in data:
        play_tennis = play_map[row[5]]
        Y.append(play_tennis)
    return Y

# Load and process training data
training_data = load_data('weather_training.csv')
X_train = encode_features(training_data)
Y_train = encode_labels(training_data)

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X_train, Y_train)

# Load and process test data
test_data = load_data('weather_test.csv')
X_test = encode_features(test_data)

#Printing the header os the solution
print("{:<6} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "Day", "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis", "Confidence"))

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
predictions = clf.predict_proba(X_test)

for i, (row, prediction) in enumerate(zip(test_data, predictions)):
    no_prob, yes_prob = prediction
    decision = 'Yes' if yes_prob > no_prob else 'No'
    confidence = yes_prob if decision == 'Yes' else no_prob
    if confidence >= 0.75:
        print("{:<6} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10.2f}".format(
            row[0],  # Day
            row[1],  # Outlook
            row[2],  # Temperature
            row[3],  # Humidity
            row[4],  # Wind
            decision,
            confidence))

