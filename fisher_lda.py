import numpy as np
from copy import deepcopy as dp
from numpy import matlib
from numpy.linalg import inv


def fisher_ld(data, num_classes, class_number):

    # Compute discriminant and project data to it.
    # Get number of classes
    col = np.size(data, 1) # get all the columns
    feat_per_class = int(col/num_classes)
    class_numbers_length = len(class_number)

    # Compute Sw and its inverse.
    Sw = np.zeros([feat_per_class,feat_per_class])
    s1 = {'1':None, '2': None, '3':None}
    XClasses = {'1':[],'2':[],'3':[] }
    YClasses = {'1':[],'2':[],'3':[] }
    mean_vect = np.zeros([feat_per_class,feat_per_class])

    mean_x1_x2_x3 = []
    for each_class in class_number:
        dict_key = str(each_class)

        #  collect all the data for the classes for which we need to compute Sw'
        col1 = int(feat_per_class)*(each_class -1)

        x1_x2_x3_feat_array = np.zeros([10,3])
        for i in range(3):
            # x_feat.append(data[:, col1 + i])
            mean_x1_x2_x3.append(np.round(np.mean(data[:, col1 + i]), 4))
            x1_x2_x3_feat_array[:, i] = np.round(data[:, col1 + i], 4)

        XClasses[dict_key].append(dp(x1_x2_x3_feat_array))

        mean_array = np.array([mean_x1_x2_x3[0], mean_x1_x2_x3[1], mean_x1_x2_x3[2]])
        mean_x1_x2_x3.clear()
        mean_vect[:, each_class-1] = np.transpose(mean_array)
        M = np.matlib.repmat(mean_array, 10,1)
        Sw = np.round((Sw + np.matmul(np.transpose(x1_x2_x3_feat_array - M), (x1_x2_x3_feat_array-M))),3)

    sw_inv = inv(Sw)
    temp_array = np.array(mean_vect[:, class_number[0]-1] - mean_vect[:, class_number[1]-1])
    w = np.round(np.matmul(sw_inv, temp_array), 4)
    for each_cls in class_number:
        feat_mat = XClasses[str(each_cls)]
        YClasses[str(each_cls)] = np.matmul(np.array(feat_mat), np.transpose(w))

    return YClasses, w, XClasses