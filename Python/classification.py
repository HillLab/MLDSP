import numpy as np 
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

# classificationCode in Matlab
def classify_dismat(dismat, alabels, folds, total):
    kf = KFold(n_splits=folds, shuffle=False)
    model_names = [SVC(kernel='linear'), SVC(kernel='poly', degree=2), SVC(kernel='rbf'), LinearDiscriminantAnalysis(), KNeighborsClassifier()]
    accuracies = np.zeros(shape=(folds, len(model_names)))
    for i in len(model_names):
        model_name = model_names[i]
        model = make_pipeline(StandardScaler(), model_name)
        k = 0
        for train_index, test_index in kf.split(dismat):
            X_train = dismat.iloc[train_index]
            X_test = dismat.iloc[test_index]
            y_train = alabels.iloc[train_index]
            y_test = alabels.iloc[test_index]
            model.fit(X_train, y_train)
            accuracies[i, k] = accuracy_score(y_test, model.predict(X_test))
            k += 1
    return accuracies

