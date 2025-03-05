#-------------------------------------------------------------------------
# AUTHOR: Tam Tran
# FILENAME: knn.py
# SPECIFICATION: compute the LOO-CV error rate for 1NN classifier sing a dataset
# FOR: CS 4210- Assignment #2 - Problem 3e
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

errors  = 0 
n = len(db)
#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    X = []
    for j in db:
       if j != i:
            X.append([float(x) for x in j[:-1]]) 


    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    Y = []
    for k in db:
        if k != i:
            Y.append(1 if k[-1] == "spam" else 0)  # Labels (spam=1, ham=0)

    #Store the test sample of this iteration in the vector testSample
    testSample = [float(x) for x in i[:-1]]
    
    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    testLabel = 1 if i[-1] == 'spam' else 0
    if class_predicted != testLabel:
        errors += 1

#Print the error rate
error_rate = errors / n
print(f"LOO-CV Error Rate: {error_rate:.2f}")