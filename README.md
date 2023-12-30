# PointCloud_Registration
Convex problem of  Point Cloud Registration

 ```
----main.py     主文件 
----Convex_solver.py 凸优化求解器
----desc_match.py 特征匹配：使用fpfh算法提取特征
----FPFH_desc.py
----utils.py 
----requirements.txt  配置环境版本，安装请一键安装：pip install requirements.txt
```


# show time!!!
如下是bunny的配准结果：
![bunny-01with02](https://github.com/xiaofulenlong/PointCloud_Registration/blob/f015232848c70cfd57170b1b2cc89e599f86404b/imgs/bunny01-02.png) ![bunny-01with03](https://github.com/xiaofulenlong/PointCloud_Registration/blob/f015232848c70cfd57170b1b2cc89e599f86404b/imgs/bunny01-03.png)

如下是room的配准结果
![room-01with02](https://github.com/xiaofulenlong/PointCloud_Registration/blob/f015232848c70cfd57170b1b2cc89e599f86404b/imgs/room01-02.png) ![room-01with03](https://github.com/xiaofulenlong/PointCloud_Registration/blob/f015232848c70cfd57170b1b2cc89e599f86404b/imgs/room01-03.png)



# 论文参考：
Horowitz M B, Matni N, Burdick J W. Convex relaxations of SE (2) and SE (3) for visual pose estimation[C]//2014 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2014: 1148-1154.
