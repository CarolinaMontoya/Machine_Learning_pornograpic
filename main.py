import os
import sys
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  

def array_in_array(scalarlist):
    return [(x,) for x in scalarlist]

X = np.array([])
y = np.array([])


data = pd.read_csv(r'Output.csv', sep=',', header=None)

seperator = ' '
for row in range(len(data)):
    temp = []
    X = np.append( X , row )
    
    b = np.array([seperator.join(data[2][row])])
    temp3 = array_in_array(b)
    y = np.append( y , temp3 )

kf = KFold(n_splits=2) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)  