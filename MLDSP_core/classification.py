"""
@Daniel
"""
from collections import defaultdict
from functools import partial
from itertools import product, combinations
from sys import stdout
from typing import Tuple, Dict, DefaultDict, List

from numpy import array, ndarray, where, unique, zeros
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from MLDSP_core.utils import uprint


# noinspection PyArgumentEqualDefault
def classify_dismat(dismat: ndarray, alabels: ndarray, folds: int,
                    log, cpus: int = 4, print_file: str = stdout
                    ) -> Tuple[float,
                               DefaultDict[str, float],
                               DefaultDict[str, ndarray],
                               DefaultDict[str, List[ndarray]],
                               Dict[str, Pipeline]]:
    """
    Supervised ML model training & k-fold cross-validation
    
    Performs supervised training of machine learning models:
    (Linear Discriminant, Linear SVM, Quadratic SVM
    and k-nearest neighbour) using the distance matrix rows
    as feature vectors and user provided metadata as class
    labels.
    
    Args:
        print_file:
        cpus:
        dismat: Pairwise distance matrix of all samples in training set
            with zero diagonal and of shape (n,n) where n sample size
        alabels: Array of class labels for supervised learning
        folds: Number of folds for cross-validation split
    Returns:
        avg_accuracy: Average of all models' mean accuracy across all folds
        mean_model_accuracies: Average of each model's accuracy across all
            folds
        aggregated_c_matrix: Sum of confusion across all folds for each model
        misclassified_idx: List of indices that were incorrectly labeled
            in cross validation for each model 
        full_model: Dictionary of each ML model with it's fitted pipeline
    """
    # matlab doesn't specify what solver it uses, orginal code used (shrinkage) gamma=0
    pipes = {
        'LinearDiscriminant': make_pipeline(LinearDiscriminantAnalysis()),
        'LinearSVM': make_pipeline(StandardScaler(), SVC(
            kernel='linear', cache_size=1000,
            decision_function_shape='ovo',probability=True)),
        'QuadSVM': make_pipeline(StandardScaler(), SVC(
            kernel='poly', degree=2, cache_size=1000,
            decision_function_shape='ovo',probability=True)),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=1, leaf_size=50, metric='euclidean',
            weights='uniform', algorithm='brute', n_jobs=cpus))
    }
    labels = unique(alabels)
    n_classes = labels.shape[0]
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
        x_train, y_train = dismat[train_idx], alabels[train_idx]
        x_test, y_test = dismat[test_idx], alabels[test_idx]
        uprint(f"Computing fold {fold+1} of {model_name}\n", print_file=print_file)
        fitted = pipe_model.fit(x_train, y_train)
        prob = fitted.predict_proba(x_test)
        # roc_auc_score requires different prob format for binary case
        if n_classes <= 2:
            prob = prob[:,1]
        prediction = fitted.predict(x_test)
        acc = balanced_accuracy_score(y_test, prediction)
        accuracies[model_name].append(acc)
        mean_model_accuracies[model_name] += acc
        auroc_score = roc_auc_score(y_test, prob, multi_class='ovo')
        log.write(f'Classification results for {model_name} fold {fold+1}:\n'
                  f'Area Under the Receiver Operating Characteristic Curve: {auroc_score}\n'
                  f'{classification_report(y_test, prediction,labels=labels)}\n')
        # uprint(f'\tAccuracy of {model_name} = {acc}\n', print_file=print_file)
        cm = confusion_matrix(y_test, prediction, labels=labels, normalize=None)
        aggregated_c_matrix[model_name] += cm
        misclassified_idx[model_name].append(where(
            y_test != prediction)[0])
        if fold == folds-1:
            full_model[model_name] = pipe_model.fit(dismat, alabels)
    # Mean accuracy value across all classifiers
    mean_model_accuracies.update((key,value/folds) for (key,value) \
                                 in mean_model_accuracies.items())
    avg_accuracy = sum(mean_model_accuracies.values()) / (len(
        mean_model_accuracies))
    uprint(f"Average accuracy over all classifiers {avg_accuracy}\n",
           print_file=print_file)

    return avg_accuracy, mean_model_accuracies, aggregated_c_matrix, \
           misclassified_idx, full_model


def calcInterclustDist(distMatrix: ndarray, labels: Tuple[str]
                       ) -> DataFrame:
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
        dist = distMatrix[where(arr == i)[0]][:, where(arr == j)[0]].ravel(
        ).mean()
        inter_dist_dict[i][i] = inter_dist_dict[j][j] = 0
        inter_dist_dict[i][j] = inter_dist_dict[j][i] = dist
    return DataFrame(inter_dist_dict)
