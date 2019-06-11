import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

bankdata = pd.read_csv("Output.csv")

X = bankdata.drop('Class', axis=1)  
y = bankdata['Class']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 

svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print("=======SVC===============")
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test,y_pred))  
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test,y_pred))
print("=============================")

print("=======RANDOM FOREST===============")
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
# predictions
rfc_predict = rfc.predict(X_test)
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())