import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from plyfile import PlyData
import json
import numpy as np 
import open3d as o3d
import matplotlib.animation as animation
import imageio
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


    rotation1 = np.array(json_data['pcd_1']['rotation'])
    rotation2 = np.array(json_data['pcd_2']['rotation'])
    rotation3 = np.array(json_data['pcd_3']['rotation'])
    
    translation1 = np.array(json_data['pcd_1']['translation'])
    translation2 = np.array(json_data['pcd_2']['translation'])
    translation3 = np.array(json_data['pcd_3']['translation'])

    point_cloud1 = plyread(pc_path1)
    point_cloud2 = plyread(pc_path2)
    point_cloud3 = plyread(pc_path3)

    select_point1 = np.array([point_cloud1[row] for row in all_idx1])
    select_point2 = np.array([point_cloud2[row] for row in all_idx2])
    select_point3 = np.array([point_cloud3[row] for row in all_idx3])
    
    return [all_idx1,all_idx2,all_idx3],[select_point1,select_point2,select_point3],[pc_path1,pc_path2,pc_path3],[point_cloud1,point_cloud2,point_cloud3],[rotation1,rotation2,rotation3],[translation1,translation2,translation3]


#读取点云数据
def plyread(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    vertex_list = [[vertex['x'], vertex['y'], vertex['z']] for vertex in vertices]
    plkdata = np.array(vertex_list)
    return plkdata
    
#============= 保存提交结果 =============
def save_result(pcd_nparrays, path,num,type):
    num = num+1
    # 保存结果为ply
    ply_path = f"{path}{type}-{num}-registered.ply"
    save_result_ply(pcd_nparrays[2], ply_path)
    # 保存结果为GIF
    GIF_path = f"{path}{type}-{num}.gif"
    # save_result_GIF(pcd_nparrays, GIF_path)



def save_result_ply(nparray, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nparray)
    o3d.io.write_point_cloud(path, pcd)

def save_result_GIF(point_collections, savePath=None):
    colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    point_cloud = o3d.geometry.PointCloud()

    for i, points in enumerate(point_collections):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(colors[i])
        point_cloud += pcd

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)

    # 每次绕y轴旋转10度，获得旋转矩阵并扩展为4*4的transform矩阵
    rotation_angle = 2
    rotation_matrix = np.array(
        [
            [np.cos(np.radians(rotation_angle)), 0, np.sin(np.radians(rotation_angle))],
            [0, 1, 0],
            [
                -np.sin(np.radians(rotation_angle)),
                0,
                np.cos(np.radians(rotation_angle)),
            ],
        ]
    )
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix

    images = []  # List to store individual frames

    # Perform the rotation directly
    for _ in range(180):  # 36 frames for a full rotation (adjust as needed)
        point_cloud.transform(transform_matrix)
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        # Capture the screen image
        image = np.asarray(vis.capture_screen_float_buffer(), dtype=np.uint8) * 255
        images.append(image)  # Convert to uint8

    if savePath != None:
        imageio.mimsave(savePath, images, fps=45)

    # Destroy the window
    vis.destroy_window()




#========== 可视化 ===============

def Visual_Pc(pcd_nparrays,window_title="PC_Regi",isVisible=True):
    def changeToPcd(nparray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(nparray)
        return pcd
    
    colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    all_pcd = o3d.geometry.PointCloud()
    for i, nparray in enumerate(pcd_nparrays):
        points = changeToPcd(nparray)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points) 

        # label = o3d.geometry.Text3D(labels[i])
        points.paint_uniform_color(colors[i])
        all_pcd += points
        
    if isVisible:
        o3d.visualization.draw_geometries([all_pcd],window_name=window_title)
        


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