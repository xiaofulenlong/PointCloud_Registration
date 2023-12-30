import numpy as np
import cvxpy as cp
from utils import *
from tqdm import tqdm
import copy
from sklearn.neighbors import NearestNeighbors


def ini_SVD_Rt(Y,X):
    """
    Input:   Y:目标点云,X:去匹配点云
    Output:  R,t 旋转向量  
    """
    Y_mean = np.mean(Y, axis=0)
    X_mean = np.mean(X, axis=0)
    Y_cor = Y - Y_mean
    X_cor = X - X_mean
    
    # 使用奇异值分解（SVD）估计旋转矩阵
    H = np.dot(X_cor.T, Y_cor)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t =  Y_mean - np.dot(R,X_mean)
    return R,t


def find_nn_corr(src, tgt):
    ''' 
    使用了最近邻搜索算法
    Input:
        - src: Source point cloud (n*3)
        - tgt: Target point cloud (n*3)
    Output: 找到了源点云src中每个点对应于目标点云tgt中的最近邻点的索引。 
        - idxs:  (n, np.array)
    '''

    ''' Way1: Sklearn'''
    if src.shape[1] != 3: src = src.T
    if tgt.shape[1] != 3: tgt = tgt.T
    
    if not isinstance(src, np.ndarray):
        src = np.asarray(src.points)    # (16384*3)
        tgt = np.asarray(tgt.points)
    
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(tgt)
    dists, idxs = neighbors.kneighbors(src)  # (16384*1), (16384*1)
    return idxs.flatten()



class ConvexRelaxationSolver:
 
    def solve(self, X, Y, max_iters=50):
        
        r = cp.Variable((3,3))
        t = cp.Variable(3)

        C = cp.bmat([
            [1 + r[0][0] + r[1][1] + r[2][2], r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]],
            [r[2][1] - r[1][2], 1 + r[0][0] - r[1][1] - r[2][2], r[1][0] + r[0][1], r[0][2] + r[2][0]],
            [r[0][2] - r[2][0], r[1][0] + r[0][1], 1 - r[0][0] + r[1][1] - r[2][2], r[2][1] + r[1][2]],
            [r[1][0] - r[0][1], r[0][2] + r[2][0], r[2][1] + r[1][2], 1 - r[0][0] - r[1][1] + r[2][2]]
        ])
        constraints = [C >> 0]

        prob = cp.Problem(
            cp.Minimize(cp.norm(( r @ X + cp.vstack([t for _ in range(X.shape[1])]).T - Y), p='fro')), 
            constraints
        )

        opt = prob.solve(solver='SCS', max_iters=max_iters, verbose=False)
        r = r.value
        t = t.value

        if np.linalg.norm(r@r.T-np.eye(3)) > 1e-3:
            u,s,vh = np.linalg.svd(r)
            r = u @ vh

        return r, t


class LinearRelaxationSolver:

    R0 = np.eye(3)
    
    def solve(self, X, Y):
        '''
        Return 
            - R0: Optimal rotation (that's iteratively optimized)
            - t : Optimal translation
        '''
        assert X.shape[0] == 3, f"X should have shape (3,N), but instead got {X.shape}"

        meanX, meanY = np.mean(X, axis=1), np.mean(Y, axis=1)
        t = meanY - meanX       
        X = (X.T + t).T

        for _ in (range(50)):
            A = -np.concatenate([self.R0 @ skew_sym(x) for x in X.T])    # (N*3, 3)
            B = np.concatenate([y- self.R0@x for x,y in zip(X.T, Y.T)])  # (N*3, )

            w = cp.Variable(3)
            prob = cp.Problem(
                cp.Minimize(cp.norm((A@w - B), p='fro')), 
                [(cp.norm2(w))<=1e-2]
            )

            try:
                prob.solve()
                w = w.value
                if w is None: break
            except Exception: break
            
            R = vec_to_rot(self.R0, w)

            if (np.linalg.norm(self.R0-R) < 1e-6):
                break
            self.R0 = R

        t = meanY - (self.R0 @ meanX)

        return self.R0, t



def ConvexSolveProblem( solver,R,t,Y,X,iters=10):
    """
    input:
        solver1,solver2:松弛器
        R,t:初始化的旋转矩阵
        Y,X:点云numpy(3,N)
        
    output:
        R,t:旋转矩阵
    """
    if Y.shape[0] != 3: Y = Y.T
    if X.shape[0] != 3: X = X.T 
    
    idx = min(X.T.shape[0],Y.T.shape[0])
    #Y:[3,n],X:[3,n]
    # 使用SVD的初值迭代 
    for _ in tqdm(range(iters)):

        X_ = ((R @ copy.deepcopy(X)).T + t).T   
        R_,t_ = solver.solve(X_ , Y , max_iters=600)
        
        if (np.linalg.norm(R_-R) < 1e-6):
            break

        R = R_ @ R              #  更新 R, t
        t = R_ @ t + t_
   
    # R,t = solver.solve(X , Y , max_iters=600)
   

    return R,t



