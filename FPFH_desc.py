import open3d as o3d
import numpy as np

def compute_fpfh_descriptor(cloud_path,Point_index,type):
    """
    input:
        str
        np.ndarray[np.float64]
    output:
         np.ndarray[np.float64]
    
    """
    def selected_hyperparameters(type):
        if type == 'bunny':
            radius_normal = 0.05  # 法向量计算时的搜索半径
            radius_feature = 0.05  # 计算 FPFH 特征时的搜索半径
            max_nn_norm = 40  # 法向量计算时的最大邻域点数
            max_nn_fpfh = 50  # 计算 FPFH 特征时的最大邻域点数
           
        elif type == 'room':
            radius_normal = 0.05   
            radius_feature = 0.05 
            max_nn_norm = 30  
            max_nn_fpfh = 50  
        
        elif type == 'temple':
            radius_normal = 0.5  
            radius_feature = 0.8  
            max_nn_norm = 50   
            max_nn_fpfh = 120    
        return radius_normal,radius_feature,max_nn_norm,max_nn_fpfh


    #读取点云数据
    pointCloud = o3d.io.read_point_cloud(cloud_path)
    # 计算 FPFH 特征
    radius_normal,radius_feature,max_nn_norm,max_nn_fpfh = selected_hyperparameters(type) #确定参数
    pointCloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_norm))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pointCloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn_fpfh)
    )

    # 获取 FPFH 特征矩阵
    fpfh_data = fpfh.data[:,Point_index]

    return fpfh_data.T
