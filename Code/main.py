import time
import sys
import itertools
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from load_data import prepare_data
from dl_model1 import dl_model

from svm import support_vector_machine
from random_forest import random_forest
from gradient_boosting import gradient_boosting
from knn import k_neighbors
from logistic_regression import logistic_regression

# main
if __name__ == '__main__':
    x_train, x_test, y_train, y_test = prepare_data()

    dl_model_time, dl_model_accuracy = dl_model(x_train, x_test, y_train, y_test)

    svm_time, svm_accuracy = support_vector_machine(x_train, x_test, y_train, y_test)

    rf_time, rf_accuracy = random_forest(x_train, x_test, y_train, y_test)

    grad_time, grad_accuracy = gradient_boosting(x_train, x_test, y_train, y_test)

    k_time, k_accuracy = k_neighbors(x_train, x_test, y_train, y_test)

    lr_time, lr_accuracy = logistic_regression(x_train, x_test, y_train, y_test)

    accuracy = [svm_accuracy, rf_accuracy, dl_model_accuracy, grad_accuracy,
                k_accuracy, lr_accuracy]
    time_ = [svm_time, rf_time, dl_model_time, grad_time, k_time, lr_time]

    plt.ylim(0, 100)
    plt.xlabel("Accuracy ")
    plt.title("Comparison of performance - Accuracy")
    l1, l2, l3, l4, l5, l6 = plt.bar(["SVM", "RF", "MLP",
                                                "GB", "KNN", "LR"],
                                                accuracy)
    
    plt.xticks(rotation=45)

    l1.set_facecolor('r')
    l2.set_facecolor('r')
    l3.set_facecolor('r')
    l4.set_facecolor('r')
    l5.set_facecolor('r')
    l6.set_facecolor('r')
    
    plt.show()
    plt.close('all')
    plt.ylim(0, 10)
    plt.xlabel("Execution time")
    plt.title("Comparison of performance (Execution Time)")
    c1, c2, c3, c4, c5, c6 = plt.bar(["SVM", "RF", "MLP",
                                                "GB", "KNN", "LR",
                                                ],
                                                time_)
    c1.set_facecolor('b')
    c2.set_facecolor('b')
    c3.set_facecolor('b')
    c4.set_facecolor('b')
    c5.set_facecolor('b')
    c6.set_facecolor('b')
    plt.xticks(rotation=45)
    plt.show()        
