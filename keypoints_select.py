import numpy as np
from sklearn.neighbors import KDTree

def select_keypoints(points,index,radius):
    """
    Input:
        points数组应该是一个表示三维点云的 NumPy 数组,其中每行包含一个点的XYZ 坐标。ndarray[N,3]
    Returns:
        包含被选为关键点的点在输入数组中的索引的一维数组坐标。ndarray。
    """
    selected = np.zeros(points.shape[0], dtype=bool)
    visited = np.zeros(points.shape[0], dtype=bool)
    kdtree = KDTree(points)

    while not visited.all():
        point_idx = (~visited).nonzero()[0][0]
        selected[point_idx] = True
        visited[kdtree.query_radius([points[point_idx]], radius)[0]] = True

    return selected.nonzero()[0]
    # return index

