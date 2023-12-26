import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
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
     
    pc_path1 = path+json_data['pcd_1']['pcd_name']
    pc_path2 = path+json_data['pcd_2']['pcd_name']
    pc_path3 = path+json_data['pcd_3']['pcd_name']

    point_cloud1 = plyread(pc_path1)
    point_cloud2 = plyread(pc_path2)
    point_cloud3 = plyread(pc_path3)

    select_point1 = np.array([point_cloud1[row] for row in all_idx1])
    select_point2 = np.array([point_cloud2[row] for row in all_idx2])
    select_point3 = np.array([point_cloud3[row] for row in all_idx3])
    
    return all_idx1,all_idx2,all_idx3,select_point1,select_point2,select_point3,pc_path1,pc_path2,pc_path3,point_cloud1,point_cloud2,point_cloud3


#读取点云数据
def plyread(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    vertex_list = [[vertex['x'], vertex['y'], vertex['z']] for vertex in vertices]
    plkdata = np.array(vertex_list)
    return plkdata
    



#========== 可视化 ===============

def Visual_Pc(pcds, labels=None, path=None):
    if labels is None: 
        labels = ['Original', 'Corrupted', 'Recovered']

    dpi = 80
    fig = plt.figure(figsize=(1440/dpi, 720/dpi), dpi=dpi)
    ax = fig.add_subplot(projection='3d')

    # ax.set_xlim3d([-3,3]), ax.set_ylim3d([-3,3]), ax.set_zlim3d([-3,3])
    ax.set_proj_type('persp')

    for points, label in zip(pcds, labels):
        ax.scatter(points[:, 0], points[:, 2], points[:, 1],
                marker='.', alpha=0.5, edgecolors='none', label=label)

    plt.legend()
    

    # plt.show()
    if path is not None:
        plt.savefig(f"/home/hrr/my_code/my_cvxpy/PointCloud_Registration/imgs/{path}.png")
    # plt.clf()
        




def skew_sym(x):    
    ''' Given a vector x, apply skew-symmetric operator '''
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])    

def vec_to_rot(R0, w):
    ''' Given a initial (3,3) rotation R0 and an (3,) angle w from R0
    compute new rotation matrix'''
    return R0 @ (np.eye(3) + skew_sym(w))

def mse(X, Y): return np.mean((X-Y)**2)