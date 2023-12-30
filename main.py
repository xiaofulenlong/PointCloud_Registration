from utils import *
from keypoints_select import *
from FPFH_desc import *
from desc_match import *
from Convex_solver import *

#路径
json_path = "/Users/hurenrong/workplace/课程课件/凸优化/期末pro/code/PointCloud_Registration/data/bunny-pcd/"

#读取点云数据
"""
idx_Y,idx_X1,idx_X2:点云1(target),点云2,点云3的索引
Y,X1,X2:点云1、2、3中提取相关索引的值,shape:(n,3)
PC_path_Y,PC_path_X1,PC_path_X2:点云地址
point_cloudY,point_cloudX1,point_cloudX2:点云元数据,优化过程中不能使用,但是可视化结果时需要
"""
idx_Y,idx_X1,idx_X2,Y,X1,X2,PC_path_Y,PC_path_X1,PC_path_X2,point_cloudY,point_cloudX1,point_cloudX2= readjson(json_path)

#选取关键点：再次筛选离群点和高噪声点
"""
input:json中要求的点云坐标 numpy[n,3]
Output:关键点索引坐标 numpy[n]
"""
idx = min(len(idx_Y),len(idx_X1))
keypoint_voxel_size = 0.005 #用于选择关键点的球形邻域的半径
keypoints_Y = select_keypoints(Y,idx_Y[:idx],keypoint_voxel_size)
keypoints_X1 = select_keypoints(X1,idx_X1[:idx],keypoint_voxel_size)


#计算特征: FPFH 
"""
input:点云数据,索引
output:FPFH特征值 numpy[n,m] n:点的数目,m:特征的数目
"""
descriptors_Y = compute_fpfh_descriptor(PC_path_Y,keypoints_Y) #[33,33]
descriptors_X1 = compute_fpfh_descriptor(PC_path_X1,keypoints_X1)


#匹配
"""
match_indices[:,0]:Y对应的坐标
match_indices[:,1]:X对应的坐标
"""
match_indices= find_descriptors_matches(descriptors_Y,descriptors_X1)

#通过凸优化方法计算R，t
    #初始化R,t,使用奇异值分解【如果采取随机初始化误差会非常之大】
R,t = ini_SVD_Rt(Y[match_indices[:,0]],X1[match_indices[:,1]])

    #松弛器
solver1 = LinearRelaxationSolver() #还算靠谱，位置不错，但是旋转的角度很差
solver2 = ConvexRelaxationSolver() #旋转角度不错
    #解决:
R,t = ConvexSolveProblem(solver1,solver2,R,t,Y[match_indices[:,0]],X1[match_indices[:,1]],iters=2) 
print(R)
print(t)

#可视化以及loss分析
# Visual_Pc([point_cloudY.T, point_cloudX1.T, (R @ point_cloudX1.T).T + t],['Y_target','X_source', 'Recovered'],f"test_X1")
res = (R @ point_cloudX1.T).T + t
Visual_Pc([point_cloudY, point_cloudX1,res])