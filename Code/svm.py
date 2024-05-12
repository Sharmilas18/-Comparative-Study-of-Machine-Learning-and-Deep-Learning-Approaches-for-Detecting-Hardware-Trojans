import time
import sys
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def support_vector_machine(train_x, test_x, train_y, test_y):
    """
    This function performs classification with support vector machine
    """

    train_y = train_y.reshape((train_y.shape[0], ))
  
    classifier = SVC(kernel="rbf", C=10, gamma=1)

    start = time.time()
    classifier.fit(train_x, train_y)
    end = time.time()

    y_pred = classifier.predict(test_x)

    time_ = end - start
    accuracy = 100 * accuracy_score(test_y, y_pred)
    
    print("### SVM ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("For C : ", 10, ", Gamma: ", 1, ", kernel = rbf",
          " => Accuracy = %.2f" % (accuracy))
        
    return(time_, accuracy)

