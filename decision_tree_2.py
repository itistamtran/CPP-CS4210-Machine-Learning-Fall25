#-------------------------------------------------------------------------
# AUTHOR: Tam Tran
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train and evaluate decision tree classifiers on contact lens data
# FOR: CS 4210- Assignment #2 - Problem 2
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

# Define the datasets and the test file
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
test_file = 'contact_lens_test.csv'

#Transform the original categorical training features to numbers and add to the 4D array X.
attribute_mappings = {
    'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3,
    'Myope': 1, 'Hypermetrope': 2,
    'Yes': 1, 'No': 2,
    'Reduced': 1, 'Normal': 2
}
#Transform the original categorical training classes to numbers and add to the vector Y.
class_mappings = {'Yes': 1, 'No': 2}

# Read and process the test data 
test_X = [] # Hold test data features
test_Y = [] # Hold test data labels
# Open and read test file, transform data using the mappings
with open(test_file, 'r') as csvfile:
    test_reader = csv.reader(csvfile)
    next(test_reader)  # Skip the header
    for row in test_reader:
        test_X.append([attribute_mappings[feature] for feature in row[:-1]])
        test_Y.append(class_mappings[row[-1]]) 

# Process each dataset
for ds in dataSets:

    X = []
    Y = []
    # Open and read training dataset file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            X.append([attribute_mappings[feature] for feature in row[:-1]])  # Transform features
            Y.append(class_mappings[row[-1]])  # Transform class
   

    if X and Y:
        accuracies = []
        for i in range(10):
            clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
            clf.fit(X, Y) #Train the model on the loaded data

            # Evaluate the classifier
            correct_predictions = sum(1 for features, true_label in zip(test_X, test_Y)
                                      if clf.predict([features])[0] == true_label)
            accuracy = correct_predictions / len(test_Y) if test_Y else 0
            accuracies.append(accuracy)

        average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        print(f'Final accuracy when training on {ds}: {average_accuracy:.2f}')
    else:
        print("Training data is empty or invalid for dataset:", ds)
