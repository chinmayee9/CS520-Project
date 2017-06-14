import sys
import numpy as np
from pre_process import readfile, get_features_for_digits, getlabels, getsamples, get_features_for_faces
from perceptron import test_multiclass_perceptron, test_binary_perceptron
from knn import test_knn
from naive_bayes import bayes


def test(type_of_data, algorithm):
    if type_of_data == 1:
        test_data = 'testimages'
        test_labels = 'testlabels'
        weights_perceptron = np.load('digits_perceptron_weights.npy')
        features_train = np.load('digits_knn_features.npy')
        height = 28
        width = 28
        classes = 10
    else:
        test_data = 'facedatatest'
        test_labels = 'facedatatestlabels'
        weights_perceptron = np.load('faces_perceptron_weights.npy')
        features_train = np.load('faces_knn_features.npy')
        height = 70
        width = 60
        classes = 2
    samples, sample_lines = readfile(test_data, type_of_data)
    samples = getsamples(samples, sample_lines, height, width)
    labels, label_lines = readfile(test_labels, type_of_data)
    labels = getlabels(labels)
    if type_of_data == 1:
        feature_matrix = get_features_for_digits(samples)
        if algorithm == 'perceptron':
            test_multiclass_perceptron(feature_matrix, labels, classes, weights_perceptron)
        if algorithm == 'knn':
            test_knn(feature_matrix, labels, features_train, 1)
        if algorithm == 'naivebayes':
            bayes(feature_matrix, labels, features_train, 1)
    else:
        feature_matrix = get_features_for_faces(samples)
        if algorithm == 'perceptron':
            test_binary_perceptron(feature_matrix, labels, classes, weights_perceptron)
        if algorithm == 'knn':
            test_knn(feature_matrix, labels, features_train, 2)
        if algorithm == 'naivebayes':
            bayes(feature_matrix, labels, features_train, 2)


if __name__ == '__main__':
    test(int(sys.argv[1]), sys.argv[2])
