import time
import sys
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def k_neighbors(train_x, test_x, train_y, test_y):
    """
    This function performs classification with k neighbors algorithm.
    """

    train_y = train_y.reshape((train_y.shape[0], ))
          
    clf = KNeighborsClassifier(n_neighbors=3)
    
    start = time.time()
    clf.fit(train_x, train_y)
    end = time.time()

    y_pred = clf.predict(test_x)

    time_ = end - start
    accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### KNN ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("Accuracy = %.2f" % (accuracy))

    return(time_, accuracy)

