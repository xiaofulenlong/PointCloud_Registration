import numpy as np
from scipy.spatial.distance import cdist


def find_descriptors_matches(descriptors_Y, descriptors_X1):
    """
    为给定的两组描述符找到最接近的匹配.

    Input:
        descriptors_Y: numpy[n,m]目标的特征描述符
        descriptors_X1: 匹配的特征描述符

    Returns:
        找到的匹配点的索引. [np.ndarray[np.int32], np.ndarray[np.int32]]
    """
    non_empty_Y = np.any(descriptors_Y, axis=1).nonzero()[0]
    non_empty_X = np.any(descriptors_X1, axis=1).nonzero()[0]
    distance_matrix = cdist(
        descriptors_Y[non_empty_Y],
        descriptors_X1[non_empty_X],
    )
    indices = distance_matrix.argmin(axis=1)
    return non_empty_Y[indices], non_empty_X[indices]
