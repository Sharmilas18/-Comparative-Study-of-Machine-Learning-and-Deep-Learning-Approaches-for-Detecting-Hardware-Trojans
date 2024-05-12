import time
import sys
import itertools
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical as npu

    
def create_model(train_x, test_y):
    num_classes = test_y.shape[1]

    model = Sequential()

    model.add(Dense(15, input_dim=train_x.shape[1], activation='relu'))

    model.add(Dense(75, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['mse', 'accuracy'])
    return model


def dl_model(train_x, test_x, train_y, test_y):

    labels = test_y

    train_y = npu(train_y)
    test_y = npu(test_y)
    
    model = create_model(train_x, test_y)
    
    start = time.time()
    model.fit(train_x, train_y, epochs=50, batch_size=10, shuffle=False)
    end = time.time()
    
    y_pred = model.predict(test_x)
    predictions = np.argmax(y_pred, axis=1)

    correct_class = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            correct_class += 1

    time_ = end - start
    accuracy = (correct_class / len(labels)) * 100

    print("### Deep Learning Model ###\n")
    print("Training lasted %.2f seconds" % time_)
    print("Accuracy = %.2f" % (accuracy))

    return(time_, accuracy)