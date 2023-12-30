import numpy as np
from scipy.spatial.distance import cdist


def find_descriptors_matches(descriptors_Y, descriptors_X,type,num_i):
    """
    为给定的两组描述符找到最接近的k个匹配点.

    Input:
        descriptors_Y: numpy[n,m]目标的特征描述符
        descriptors_X1: 匹配的特征描述符

    Returns:
        match_point:找到的匹配点的索引. (top, 2)
        每一行包含两个整数,分别代表最近邻的点在descriptors_Y和descriptors_X中的索引。 
        
    """
    #匹配对的最大数目
    def selected_points_num(type):
        if type == 'bunny':
            if num_i == 1:
                top = 8
            else:
                top = 5
        elif type == 'room':
            top = 5
        elif type == 'temple':
            top = 6
        else:
            top = min(descriptors_Y.shape[0],descriptors_X.shape[0])
        return top
    top = selected_points_num(type)
    #降噪,除去零向量 【暂时不需要，因为在temple中可用点 <<<< 点云数目，去除0向量之后猛猛报错】
    # non_empty_Y = np.any(descriptors_Y, axis=1).nonzero()[0]
    # non_empty_X = np.any(descriptors_X, axis=1).nonzero()[0]
    distance_matrix = cdist(
        descriptors_Y,
        descriptors_X
    )
    match_point = []
    #开始配对:最近邻匹配
    for _ in range(top):
        #查找最小距离
        idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        #匹配
        match_point.append((idx[0], idx[1]))
        distance_matrix[idx[0], :] = np.inf
        distance_matrix[:, idx[1]] = np.inf

    return  np.asarray(match_point)
