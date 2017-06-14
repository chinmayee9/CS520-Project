import numpy as np
from scipy.spatial.distance import cdist
from pre_process import readfile, getlabels
from collections import Counter


def test_knn(features, y_expected, knn_features, type_of_data):
    if type_of_data == 1:
        labels, label_lines = readfile('traininglabels', 1)
        k = 3
    else:
        labels, label_lines = readfile('facedatatrainlabels', 2)
        k = 1
    labels = getlabels(labels)
    D = cdist(features, knn_features)
    D_index = np.argsort(D, axis=1)
    nearestNeighbour = D_index[:, 0:k]
    Ypred = []
    for i in range(len(nearestNeighbour)):
        temp = []
        for j in range(k):
            temp.append(labels[nearestNeighbour[i, j]])
        Ypred.append(temp)
    newYpred = []
    for row in Ypred:
        cnt = Counter(row).most_common(1)
        newYpred.append(cnt[0][0])
    error = 0
    for i in range(len(y_expected)):
        print 'expected = ', y_expected[i], 'predicted = ', newYpred[i]
        if newYpred[i] == y_expected[i]:
            continue
        else:
            error += 1
    print "Prediction accuracy = ", (1 - error / float(len(y_expected))) * 100, "%"
