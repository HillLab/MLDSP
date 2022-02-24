"""
@Daniel
"""
from collections import defaultdict
from functools import partial

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def classify_dismat(dismat, alabels, folds):
    """
    @Daniel
    Args:
        dismat: 
        alabels:
        folds:

    Returns:

    """
    # matlab doesn't specify what solver it uses, orginal code used (shrinkage) gamma=0
    pipes = {
        'LinearDiscriminant': make_pipeline(LinearDiscriminantAnalysis()),
        'LinearSVM': make_pipeline(StandardScaler(), SVC(
            kernel='linear', cache_size=1000,
            decision_function_shape='ovo')),
        'QuadSVM': make_pipeline(StandardScaler(), SVC(
            kernel='poly', degree=2, cache_size=1000,
            decision_function_shape='ovo')),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=1, leaf_size=50, metric='euclidean',
            weights='uniform', algorithm='brute'))
    }
    n_classes = np.unique(alabels).shape[0]
    accuracies = defaultdict(list)
    conf_matrix_dict = defaultdict(list)
    misclassified_idx = defaultdict(list)
    mean_model_accuracies = defaultdict(float)
    aggregated_c_matrix = defaultdict(partial(np.zeros,
                                              (n_classes, n_classes)))
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=23)
    best_model = defaultdict(lambda: (Pipeline, 0.0))

    for i, (train_index, test_index) in enumerate(kf.split(
            dismat, alabels)):
        x_train, y_train = dismat[train_index], alabels[train_index]
        x_test, y_test = dismat[test_index], alabels[test_index]
        print(f"Computing fold {i+1}")
        for model_name, pipe_model in pipes.items():
            pipe_model.fit(x_train, y_train)
            prediction = pipe_model.predict(x_test)
            acc = accuracy_score(y_test, prediction)
            if best_model[model_name][1] < acc:
                best_model[model_name] = (pipe_model, acc)
            accuracies[model_name].append(acc)
            mean_model_accuracies[model_name] += acc
            print(f'\tAccuracy of {model_name} = {acc}')
            cm = confusion_matrix(y_test, prediction,
                                  labels=list(np.unique(alabels)),
                                  normalize=None)
            conf_matrix_dict[model_name].append(cm)
            aggregated_c_matrix[model_name] += cm
            # Store indices (of dismat) of misclassified sequences
            misclassified_idx[model_name].append(np.where(
                y_test == prediction)[0])

    # Mean accuracy value across all classifiers
    avg_accuracy = sum(mean_model_accuracies.values()) / (folds * len(
        mean_model_accuracies))
    print(f"Average accuracy over all classifiers {avg_accuracy}")

    return avg_accuracy, mean_model_accuracies, aggregated_c_matrix, \
           misclassified_idx, best_model


def displayConfusionMatrix(confMatrix, alabels):
    """
    @Daniel
    Args:
        confMatrix:
        alabels:

    Returns:

    """
    # generate cm image and plot
    confMatrixDisplayObj = ConfusionMatrixDisplay(
        confusion_matrix=confMatrix, display_labels=list(np.unique(alabels)))
    confMatrixDisplayObj.plot(cmap='Blues', colorbar=False)
    return confMatrixDisplayObj
