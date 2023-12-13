import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


#loading data
data=load_breast_cancer()
X = data.data
y = data.target
#split data with 10-fold cross validation.
kf = KFold(n_splits=10)


#build Support vector classifier
svc = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None) 


#build Decision Tree classifier
clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2)

#classification performance of SVC
acc=[]
f1=[]
auc_roc=[]
for train_index, test_index in kf.split(X, y):
    #training and test feature data n labels
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #fits SVC model on training data
    svc.fit(X_train, y_train)
    
    #predictions on X_test
    y_pred_svc = svc.predict(X_test)
    
    acc.append(accuracy_score(y_test, y_pred_svc))
    f1.append(f1_score(y_test, y_pred_svc))
    auc_roc.append(roc_auc_score(y_test, y_pred_svc))
    
print(np.mean(acc))
print(np.mean(f1))
print(np.mean(auc_roc))



#classification performance of Decision Tree classifier
acc=[]
f1=[]
auc_roc=[]
for train_index, test_index in kf.split(X, y):
    #training and test feature data n labels
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #fits Decision Tree classifier model on training data
    clf.fit(X_train, y_train)
    
    #predictions on X_test
    y_pred_clf = clf.predict(X_test)
    
    acc.append(accuracy_score(y_test, y_pred_clf))
    f1.append(f1_score(y_test, y_pred_clf))
    auc_roc.append(roc_auc_score(y_test, y_pred_clf))
    
print(np.mean(acc))
print(np.mean(f1))
print(np.mean(auc_roc))