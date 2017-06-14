from getfeatures import *


def readfile(name, type_of_data):
    if type_of_data == 1:
        file = open('./data/digitdata/%s' % name, 'r')
    else:
        file = open('./data/facedata/%s' % name, 'r')
    lines = file.readlines()
    return lines, len(lines)


def getsamples(samples, sample_lines, height, width):
    sample_array = []
    for line in samples:
        for i in range(width):
            if line[i] == ' ':
                sample_array.append(0)
            elif line[i] == '#':
                sample_array.append(255)
            elif line[i] == '+':
                sample_array.append(225)
            else:
                continue
    sample_array = np.array(sample_array)
    sample_array = sample_array.reshape(sample_lines / height, height, width)
    return sample_array


def getlabels(labels):
    result = []
    for label in labels:
        result.append(int(label[0]))
    return np.array(result)


def get_features_for_digits(samples):
    white_pixels = getwhitepixels(samples)
    r = len(white_pixels)
    white_pixels = np.array(white_pixels).reshape(r, 1)
    white_pixels_row_wise = np.array(getwhitepixels_byrows(samples))
    white_pixels_col_wise = np.array(getwhitepixels_bycols(samples))
    product_row_col = white_pixels_row_wise * white_pixels_col_wise
    window_pixel_count = np.array(get_window_pixels(samples, 7))
    hu_moments = np.array(get_hu_moments(samples))
    features = np.concatenate((white_pixels, white_pixels_row_wise, white_pixels_col_wise,
                               window_pixel_count, hu_moments), axis=1)
    return features


def get_features_for_faces(samples):
    white_pixels = getwhitepixels(samples)
    r = len(white_pixels)
    white_pixels = np.array(white_pixels).reshape(r, 1)
    white_pixels_row_wise = np.array(getwhitepixels_byrows(samples))
    white_pixels_col_wise = np.array(getwhitepixels_bycols(samples))
    window_pixel_count = get_window_pixels(samples, 10)
    features = np.concatenate((white_pixels, white_pixels_row_wise,
                               white_pixels_col_wise, window_pixel_count), axis=1)
    return features
