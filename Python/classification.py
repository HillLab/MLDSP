import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


def classify_dismat(dismat, alabels, folds, total):
    """ Keyword arguments:
        Required: folds, total
        Optional: dismat, alabels, test_set_distances
    """
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=23)
    model_names = [LinearDiscriminantAnalysis(), SVC(kernel='linear'), SVC(
        kernel='poly', degree=2),  KNeighborsClassifier(), SVC(kernel='rbf')]
    accuracies = np.zeros(shape=(len(model_names), folds))
    average_accuracy = []
    average_accuracy += [model_names]
    for i in range(len(model_names)):
        model_name = model_names[i]
        # create a pipeline object
        model = make_pipeline(StandardScaler(), model_name)
        k = 0
        for train_index, test_index in kf.split(dismat, alabels):
            X_train = dismat[train_index]
            X_test = dismat[test_index]
            y_train = [alabels[i] for i in train_index]
            y_test = [alabels[i] for i in test_index]
        # fit the whole pipeline
            model.fit(X_train, y_train)
            accuracies[i, k] = accuracy_score(y_test, model.predict(X_test))
            k += 1
        # Work in progress, will probably rewrite entire classification pipeline
        # plot_confusion_matrix(model, X_test, y_test)
        # plt.show()
        # future implementation of testing
        # if test_set!=None:
        #     model.predict()
    # Mean of each classifier across 10 folds
    mean_accuracy = accuracies[:, 1:].mean(axis=1)*100
    # Mean accuracies for all classifiers
    average_accuracy.append(mean_accuracy.tolist())
    return mean_accuracy, average_accuracy
