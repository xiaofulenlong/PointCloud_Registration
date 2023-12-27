import numpy as np
import open3d as o3d


def compute_fpfh_descriptor(cloud_path,Point_index):
    """
    input:
        str
        np.ndarray[np.float64]
    output:
         np.ndarray[np.float64]
    
    """
    #读取点云数据
    pointCloud = o3d.io.read_point_cloud(cloud_path)
    # 计算 FPFH 特征
    radius_normal = 0.1  # 法向量计算时的搜索半径
    radius_feature = 0.3  # 计算 FPFH 特征时的搜索半径

    pointCloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pointCloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    # 获取 FPFH 特征矩阵
    fpfh_data = np.asarray(fpfh.data[:,Point_index])

    return fpfh_data.T
