import time
import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def random_forest(train_x, test_x, train_y, test_y):
    """
    This function performs classification with random forest.
    """
    train_y = train_y.reshape((train_y.shape[0], ))
        
    clf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=1)
    
    start = time.time()
    clf.fit(train_x, train_y)    
    end = time.time()

    y_pred = clf.predict(test_x)

    time_ = end - start
    accuracy = 100 * accuracy_score(test_y, y_pred)

    print("### RF ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("Accuracy = %.2f" % (accuracy))
    print("F1-score = ",f1_score(test_y, y_pred, average='macro')*100)
        
    return(time_, accuracy)
