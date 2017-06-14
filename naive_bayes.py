import numpy as np
from pre_process import readfile, getlabels
from collections import Counter


def bayes(features, y_expected, bayes_features, type_of_data):
    if type_of_data == 1:
        labels, label_lines = readfile('traininglabels', 1)
        classes = 10
    else:
        labels, label_lines = readfile('facedatatrainlabels', 2)
        classes = 2
    labels = getlabels(labels)
    prob_y = Counter(labels)
    prob_y = sorted(prob_y.items())
    prob_y = np.array([x[1] for x in prob_y])
    prob_y = prob_y / float(len(labels))
    prob_y = np.log(prob_y)
    test_images, feature_length = features.shape
    y_predicted = []
    for t in range(test_images):
        test_feature = features[t]
        feature_sum = np.zeros(classes, dtype=np.float)
        for i in range(feature_length):
            count = classes * [0]
            total_instances_y = classes * [0]
            for j in range(len(labels)):
                for k in range(classes):
                    if labels[j] == k:
                        total_instances_y[k] += 1
                        if test_feature[i] == bayes_features[j, i]:
                            count[k] += 1
            count = [1 if c == 0 else c for c in count]
            count = np.array(count)
            total_instances_y = np.array(total_instances_y)
            prob_fi = count / total_instances_y.astype(dtype=np.float)
            prob_fi = np.log(prob_fi)
            feature_sum = feature_sum + prob_fi
        total = prob_y + feature_sum
        y_predicted.append(np.argmax(total))
        print "predicted = ", y_predicted[t], "expected = ", y_expected[t]
    error = 0
    for i in range(test_images):
        if y_predicted[i] == y_expected[i]:
            continue
        else:
            error += 1
    print "Prediction accuracy = ", (1 - error / float(test_images)) * 100, "%"
