import time
import sys
import numpy as np

from confusion_matrix1 import plot_confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def gradient_boosting(train_x, test_x, train_y, test_y):
    """
    This function performs classification with Gradient Boosting.
    """

    train_y = train_y.reshape((train_y.shape[0], ))
        
    clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=75)

    start = time.time()
    clf.fit(train_x, train_y)    
    end = time.time()

    y_pred = clf.predict(test_x)

    time_ = end - start
    accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### GB ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("Accuracy = %.2f" % (accuracy))

    
    conf_matrix = confusion_matrix(test_y, y_pred)
    plot_confusion_matrix(cm=conf_matrix, target_names=['Trojan Free','Trojan Infected'], title='Confusion matrix')
    
    return(time_, accuracy)

