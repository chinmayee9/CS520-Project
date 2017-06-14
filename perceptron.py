import numpy as np


def train_multiclass_perceptron(features, y_expected, classes):
    r, c = features.shape
    weights = np.random.rand(c, classes)
    epochs = 0
    while epochs < 1000:
        error = 0
        for i in range(r):
            feature_vector = features[i, ::]
            y_predicted = np.argmax(np.dot(feature_vector, weights))
            if y_predicted == y_expected[i]:
                continue
            else:
                error += 1
                weights[::, y_expected[i]] = weights[::, y_expected[i]] + feature_vector.transpose()
                weights[::, y_predicted] = weights[::, y_predicted] - feature_vector.transpose()
        epochs += 1
        print error / float(r)
    return weights


def train_binary_perceptron(features, y_expected):
    y_expected = [y if y == 1 else -1 for y in y_expected]
    r, c = features.shape
    weights = np.random.rand(c, 1)
    epochs = 0
    while epochs < 1000:
        error = 0
        for i in range(r):
            feature_vector = features[i, ::]
            y_predicted = np.dot(feature_vector, weights)
            y_predicted = y_predicted[0]
            if y_predicted >= 0:
                y_predicted = 1
            else:
                y_predicted = -1
            if y_predicted == y_expected[i]:
                continue
            else:
                error += 1
                temp = y_expected[i] * feature_vector.transpose()
                temp = temp.reshape(c, 1)
                weights = weights + temp
        epochs += 1
        print error / float(r)
    return weights


def test_multiclass_perceptron(features, y_expected, classes, weights):
    r, c = features.shape
    error = 0
    for i in range(r):
        feature_vector = features[i, ::]
        y_predicted = np.argmax(np.dot(feature_vector, weights))
        print "expected = ", y_expected[i], "predicted = ", y_predicted
        if y_predicted == y_expected[i]:
            continue
        else:
            error += 1
    print "Prediction accuracy = ", (1 - error / float(r)) * 100, "%"


def test_binary_perceptron(features, y_expected, classes, weights):
    r, c = features.shape
    error = 0
    for i in range(r):
        feature_vector = features[i, ::]
        y_predicted = np.dot(feature_vector, weights)
        y_predicted = y_predicted[0]
        if y_predicted >= 0:
            y_predicted = 1
        else:
            y_predicted = -1
        if y_predicted == -1:
            y_predicted = 0
        print "expected = ", y_expected[i], "predicted = ", y_predicted
        if y_predicted == y_expected[i]:
            continue
        else:
            error += 1
    print "Prediction accuracy = ", (1 - error / float(r)) * 100, "%"
