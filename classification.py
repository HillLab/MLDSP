import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict

def classify_dismat(dismat, alabels, folds, total, saveModels=False):

    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=23)
    model_names = {'LinearDiscriminant':LinearDiscriminantAnalysis() #matlab doesn't specify what solver it uses, orginal code used (shrinkage) gamma=0 
                   ,'LinearSVM':SVC(kernel='linear',cache_size=1000,decision_function_shape='ovo'), 'QuadSVM':SVC(kernel='poly', degree=2,cache_size=1000,decision_function_shape='ovo'),
                   'KNN':KNeighborsClassifier(n_neighbors=1, leaf_size=50, metric='euclidean', weights='uniform', algorithm='brute')
                   }
    # use 2 additional classifiers if <= 2000 sequences
    # if total <= 2000:
    #     model_names['SubspaceDiscriminant'] = SVC(kernel='rbf')
    #     model_names['SubspaceKNN'] = None

    accuracies = defaultdict(list) # dictionary with key: modelname, value: list containing accuracies
    confMatrixDict = defaultdict(list) # dictionary with key: modelname, value: list containing confusion matrix displays
    misclassifiedIdx = defaultdict(list) # dictionary with key: modelname, value: list containing indices/sequences of dismat that have been misclassifed

    # Loop through each model
    for modelName in model_names:
        model = model_names.get(modelName)
        print(model)
        # Create pipeline model
        if modelName in ['LinearSVM', 'QuadSVM', 'KNN']:
            pipeModel = make_pipeline(StandardScaler(), model)
        else:
            pipeModel = make_pipeline(model)
            
        i =0
        for train_index, test_index in kf.split(dismat, alabels):
            i += 1
            X_train = dismat[train_index]
            X_test = dismat[test_index]
            y_train = [alabels[i] for i in train_index]
            y_test = [alabels[i] for i in test_index]

            # Fit the pipeline model
            pipeModel.fit(X_train, y_train)
            prediction = pipeModel.predict(X_test)
            # Compute and store accuracy of model
            accuracies[modelName].append(accuracy_score(y_test, prediction))
            print(accuracy_score(y_test, prediction))
            # Generate and store confusion matrix
            cm = confusion_matrix(y_test, prediction, labels=list(np.unique(alabels)), normalize=None)
            confMatrixDict[modelName].append(cm)

            # Store indices (of dismat) of misclassified sequences
            for i in range(len(prediction)):
                # if prediction incorrect, add to list of misclassified indices for the model
                if prediction[i] != y_test[i]:
                    misclassifiedIdx[modelName].append(test_index[i])
            print(i)

    # For each model, Calculate mean of accuracies across 10 folds & Sum all confusion matrices across 10 folds
    meanModelAccuracies = {} # key: modelName, value: mean accuracy value for model
    aggregatedCMatrix = {} # key: modelName, value: summed Confusion Matrix for model
    for modelName in accuracies:
        meanModelAccuracies[modelName] = np.mean(accuracies.get(modelName))
        aggregatedCMatrix[modelName] = np.sum(confMatrixDict.get(modelName), axis=0)

    # Mean accuracy value across all classifiers
    avgAccuracy = sum(meanModelAccuracies.values()) / len(meanModelAccuracies)

    return avgAccuracy, meanModelAccuracies, aggregatedCMatrix, dict(misclassifiedIdx)

# Plots and returns a ConfusionMatrix Display object from a raw array
def displayConfusionMatrix(confMatrix, alabels):
    # generate cm image and plot
    confMatrixDisplayObj = ConfusionMatrixDisplay(confusion_matrix=confMatrix, display_labels=list(np.unique(alabels)))
    confMatrixDisplayObj.plot(cmap='Blues', colorbar= False)

    # access raw cm array: cm_disp.confusion_matrix
    # alternative to display: cm_disp = plot_confusion_matrix(pipeModel, X_test, y_test, normalize=None, cmap='Blues', colorbar= False)

    return confMatrixDisplayObj
