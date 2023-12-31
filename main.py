from utils import *
from FPFH_desc import *
from desc_match import *
from Convex_solver import *

def PC_registrate(type,path):
    json_path = f"{path}data/{type}-pcd/"
    result_path  = f"{path}result/{type}/"
    #读取点云数据
    """
    idx: idx_Y,idx_X1,idx_X2:点云1(target),点云2,点云3的索引
    Selected:Y,X1,X2:点云1、2、3中提取相关索引的值,shape:(n,3)
    PC_path:PC_path_Y,PC_path_X1,PC_path_X2:点云地址
    point_cloud:point_cloudY,point_cloudX1,point_cloudX2:点云元数据,优化过程中不能使用,但是可视化结果时需要
    rotation,translation: 旋转矩阵,平移矩阵的gt值
    """
    # idx_Y,idx_X1,idx_X2,Y,X1,X2,PC_path_Y,PC_path_X1,PC_path_X2,point_cloudY,point_cloudX1,point_cloudX2= readjson(json_path)
    idx,Selected,PC_path,point_cloud,rotation,translation = readjson(json_path)
    for i in [1, 2]:
        print("第{}只{}匹配".format(i,type))
        #选取关键点：再次筛选离群点和高噪声点
        """
        input:json中要求的点云坐标 numpy[n,3]
        Output:关键点索引坐标 numpy[n]
        """
        # idx = min(len(idx_Y),len(idx_X1))
        # keypoint_voxel_size = 0.005 #用于选择关键点的球形邻域的半径
        # keypoints_Y = select_keypoints(Y,idx_Y[:idx],keypoint_voxel_size)
        # keypoints_X1 = select_keypoints(X1,idx_X1[:idx],keypoint_voxel_size)


        #计算特征: FPFH 
        """
        input:点云数据,索引
        output:FPFH特征值 numpy[n,m] n:点的数目,m:特征的数目
        """
        descriptors_Y = compute_fpfh_descriptor(PC_path[0],idx[0],type)  
        descriptors_X = compute_fpfh_descriptor(PC_path[i],idx[i],type)
        
        #匹配
        """
        match_indices[:,0]:Y对应的坐标
        match_indices[:,1]:X对应的坐标
        """

        match_indices= find_descriptors_matches(descriptors_Y,descriptors_X,type,i)
       
        #通过凸优化方法计算R，t
            #初始化R,t,使用奇异值分解【如果采取随机初始化误差会非常之大】
        R,t = ini_SVD_Rt(Selected[0][match_indices[:,0]],Selected[i][match_indices[:,1]])

            #松弛器
        # solver1 = LinearRelaxationSolver() #还算靠谱，位置不错，但是旋转的角度很差。算了，一拖拉机
        solver = ConvexRelaxationSolver() #旋转角度不错
            #解决:
        R,t = ConvexSolveProblem(solver,R,t,Selected[0][match_indices[:,0]],Selected[i][match_indices[:,1]],iters=10) 

        print(R)
        print(t)

        #可视化以及loss分析
        res = (R @ point_cloud[i].T).T + t
        l = min(res.shape[0],point_cloud[0].shape[0])
        print(f'PointCloudMSE: {mse(res[:l,:],point_cloud[0][:l,:])}')
        mse_R = np.mean((R - rotation[i]) ** 2)
        mse_t = np.mean((t - translation[i]) ** 2)
        print(f'R_and_t_MSE: {mse_R , mse_t}')
       
        save_result([point_cloud[0], point_cloud[i],res],result_path,i ,type)
        Visual_Pc([point_cloud[0], point_cloud[i],res],"第{}只{}匹配".format(i,type))





if __name__ == "__main__":
       #路径
    # json_path = "/Users/hurenrong/workplace/课程课件/凸优化/期末pro/code/PointCloud_Registration/data/bunny-pcd/"
    # json_path = "/Users/hurenrong/workplace/课程课件/凸优化/期末pro/code/PointCloud_Registration/data/room-pcd/"
    # json_path = "/Users/hurenrong/workplace/课程课件/凸优化/期末pro/code/PointCloud_Registration/data/temple-pcd/"
    i = 0 # 0:bunny 1:room 2:temple
    types = ['bunny','room','temple']
    path = "/Users/hurenrong/workplace/课程课件/凸优化/期末pro/code/PointCloud_Registration/"
    PC_registrate(types[i],path)