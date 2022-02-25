"""
@Daniel
"""
from collections import defaultdict
from functools import partial
from itertools import product, combinations
from typing import Tuple, Dict, DefaultDict, List

from numpy import array, ndarray, where, unique, zeros
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# noinspection PyArgumentEqualDefault
def classify_dismat(dismat: ndarray, alabels: ndarray, folds: int,
                    cpus: int = 4) -> Tuple[float,
                                            DefaultDict[str, float],
                                            DefaultDict[str, ndarray],
                                            DefaultDict[str, List[ndarray]],
                                            Dict[str, Pipeline]]:
    """
    @Daniel
    Args:
        cpus:
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
            weights='uniform', algorithm='brute', n_jobs=cpus))
    }
    n_classes = unique(alabels).shape[0]
    accuracies = defaultdict(list)
    misclassified_idx = defaultdict(list)
    mean_model_accuracies = defaultdict(float)
    aggregated_c_matrix = defaultdict(partial(zeros, (
        n_classes, n_classes)))
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=23)
    full_model = {}
    prod = product(pipes.items(), enumerate(kf.split(dismat, alabels)))
    for item in prod:
        (model_name, pipe_model), (fold, (train_idx, test_idx)) = item
        full_model[model_name] = pipe_model.fit(dismat, alabels)
        x_train, y_train = dismat[train_idx], alabels[train_idx]
        x_test, y_test = dismat[test_idx], alabels[test_idx]
        print(f"Computing fold {fold}")
        fitted = pipe_model.fit(x_train, y_train)
        prediction = fitted.predict(x_test)
        acc = accuracy_score(y_test, prediction)
        accuracies[model_name].append(acc)
        mean_model_accuracies[model_name] += acc
        print(f'\tAccuracy of {model_name} = {acc}')
        cm = confusion_matrix(y_test, prediction, labels=list(unique(
            alabels)), normalize=None)
        aggregated_c_matrix[model_name] += cm
        misclassified_idx[model_name].append(where(y_test == prediction
                                                   )[0])

    # Mean accuracy value across all classifiers
    avg_accuracy = sum(mean_model_accuracies.values()) / (folds * len(
        mean_model_accuracies))
    print(f"Average accuracy over all classifiers {avg_accuracy}")

    return avg_accuracy, mean_model_accuracies, aggregated_c_matrix, \
           misclassified_idx, full_model


def calcInterclustDist(distMatrix: ndarray, labels: Tuple[str]) -> str:
    """
    @ Daniel
    Args:
        distMatrix:
        labels:

    Returns:

    """
    arr = array(labels)
    un_labels = unique(arr)
    inter_dist_dict = defaultdict(dict)
    for i, j in combinations(un_labels, 2):
        dist = distMatrix[where(arr == i)[0]][:, where(arr == j)[0]
               ].ravel().mean()
        inter_dist_dict[i][i] = inter_dist_dict[j][j] = 0
        inter_dist_dict[i][j] = inter_dist_dict[j][i] = dist

    return DataFrame(inter_dist_dict).to_html()
