import numpy as np 
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler 

# classificationCode in Matlab
def classify_dismat(dismat, alabels, folds, total):
    kf = KFold(n_splits=folds, shuffle=False)
    model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5)) #todo: more models
    for train_index, test_index in kf.split(dismat):
        X_train = dismat.iloc[train_index]
        X_test = dismat.iloc[test_index]
        y_train = alabels.iloc[train_index]
        y_test = alabels.iloc[test_index]
        model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

