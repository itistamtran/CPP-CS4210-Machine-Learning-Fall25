#-------------------------------------------------------------------------
# AUTHOR: Tam Tran
# FILENAME: decision_tree.py
# SPECIFICATION: trains a decision tree classifier on this data to predict contact lens prescription, and visualizes the decision tree
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2 days
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries

import csv
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt

db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
for row in db:
    temp = []
        #Mapping Age
    if row[0] == 'Young':
        temp.append(1)
    elif row[0] == 'Prepresbyopic':
        temp.append(2)
    elif row[0] == 'Presbyopic':
        temp.append(3)
    #Mapping Spectacle Prescription
    if row[1] == 'Myope':
        temp.append(1)
    elif row[1] == 'Hypermetrope':
        temp.append(2)
    #Mapping Astigmatism
    if row[2] == 'No':
        temp.append(1)
    elif row[2] == 'Yes':
        temp.append(2)
    #Mapping Tear Prodcution Rate
    if row[3] == 'Reduced':
        temp.append(1)
    elif row[3] == 'Normal':
        temp.append(2)

# X =
    # Adding the transformed row to X
    X.append(temp)
#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
# Y =
for row in db:
    if row[4] == 'Yes':
        Y.append(1)
    elif row[4] == 'No':
        Y.append(2)
#fitting the decision tree to the data
clf = DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()