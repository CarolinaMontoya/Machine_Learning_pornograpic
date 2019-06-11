# importing required libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn import tree

clf = tree.DecisionTreeClassifier()
# reading csv file and extracting class column to y. 
#x = pd.read_csv("Output.csv") 
#a = np.array(x)
#y  = a[2] # classes having 0 and 1 
  
# extracting two features 
#x = np.column_stack((x.n)) 
#x.shape # 569 samples and 2 features 
  
dataset = pd.read_csv('Output.csv')
X = dataset.iloc[:, -1].values # Here first : means fetch all rows :-1 means except last column

Y = dataset.iloc[:, 3].values # : is fetch all rows 3 means 3rd column

# random_state below is a metric that is used by the function to shuffle datas while splitting. If you change the random_state

# then your split may not be same as previous

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 2) # 0.2 test_size means 20%

#clf = clf.fit(X_test, y_test)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)

from sklearn.model_selection import KFold # import KFold

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]]) # create an 2d array i.e independent variables matrix

Y = np.array([1, 2, 3, 4]) # Create another array i.e dependent vector



kf = KFold(n_splits=2) # Define the split - into 2 folds 

kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator

#prediction= clf.prediction(X_train, X_train)

#print(prediction)

#print(kf) 

KFold(n_splits=2, random_state=None, shuffle=False)

