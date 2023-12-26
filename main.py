from utils import *
from keypoints_select import *
from FPFH_desc import *
from desc_match import *
from Convex_solver import *
#路径
json_path = "/home/hrr/my_code/my_cvxpy/PointCloud_Registration/data/bunny-pcd/"

#读取点云数据
"""
idx_Y,idx_X1,idx_X2:点云1(target),点云2,点云3的索引
Y,X1,X2:点云1、2、3中提取相关索引的值,shape:(n,3)
PC_Y,PC_X1,PC_X2:点云元数据
"""
idx_Y,idx_X1,idx_X2,Y,X1,X2,PC_Y,PC_X1,PC_X2= readjson(json_path)

#选取关键点：再次筛选离群点和高噪声点
"""

"""
keypoint_voxel_size = 0.005 #用于选择关键点的球形邻域的半径
keypoints_Y = select_keypoints(Y,keypoint_voxel_size)
keypoints_X1 = select_keypoints(X1,keypoint_voxel_size)

#计算特征FPFH算法

descriptors_Y = compute_fpfh_descriptor(keypoints_Y)
descriptors_X1 = compute_fpfh_descriptor(keypoints_X1)


#特征匹配
[Y_indices,X1_indices]= find_descriptors_matches(descriptors_Y,descriptors_X1)

#通过凸优化方法计算R，t
    #初始化R,t,使用奇异值分解【如果采取随机初始化误差会非常之大】
R,t = ini_SVD_Rt()

    #松弛器
solver1 = LinearRelaxationSolver() #还算靠谱，位置不错，但是旋转的角度很差
solver2 = ConvexRelaxationSolver() #旋转角度不错
R,t = ConvexSolveProblem(solver1,solver2,R,t,) 


#可视化以及loss分析
Visual_Pc([Y.T, X1.T, (R @ X1).T + t],['Original', 'Corrupted', 'Recovered'],f"test_X2")
