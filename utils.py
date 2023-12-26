from plyfile import PlyData
import json
import numpy as np 

#=========== load data =============
#读取json中所需数据
def readjson(path):
    json_path = path+"points.json"
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    #选择显著点
    all_idx1 = json_data['pcd_1']['all_idxs']
    all_idx2 = json_data['pcd_2']['all_idxs']
    all_idx3 = json_data['pcd_3']['all_idxs']

    #读取点云
     
    pc_path1 = json_data['pcd_1']['pcd_name']
    pc_path2 = json_data['pcd_2']['pcd_name']
    pc_path3 = json_data['pcd_3']['pcd_name']

    point_cloud1 = plyread(path+pc_path1)
    point_cloud2 = plyread(path+pc_path2)
    point_cloud3 = plyread(path+pc_path3)

    select_point1 = np.array([point_cloud1[row] for row in all_idx1])
    select_point2 = np.array([point_cloud2[row] for row in all_idx2])
    select_point3 = np.array([point_cloud3[row] for row in all_idx3])
    
    return all_idx1,all_idx2,all_idx3,select_point1,select_point2,select_point3,point_cloud1,point_cloud2,point_cloud3


#读取点云数据
def plyread(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    vertex_list = [[vertex['x'], vertex['y'], vertex['z']] for vertex in vertices]
    plkdata = np.array(vertex_list)
    return plkdata
    



#========== 可视化 ===============
def Visual_Pc():
    return