import numpy as np
import cv2 as cv
import os
# import calib
import icp
import argparse
import random
from matplotlib import pyplot as plt
import imageio

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

import sys
# import open3d
from copy import deepcopy



def skew_matrix(array):
    x, y, z = array[0], array[1], array[2]
    return np.array([[0, -z, y],[z, 0, -x],[-y, x, 0]])

def orbfeature(img1):
    sift = cv.xfeatures2d.SIFT_create()
    kp, des1 = sift.detectAndCompute(img1,None)
    kplist=[]
    for i in range(len(kp)):
        kplist.append([kp[i].pt[1], kp[i].pt[0]])
    pts=np.array(kplist)
    return pts, kp, des1

# def featureMatching(kp1, kp2, des1, des2):
#     bf = cv.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)
#     points = []
#     pts1 = []
#     pts2 = []
#     for m, n in matches:    
#         if m.distance < 0.55 * n.distance:   
#             points.append([m])  
#             pts1.append(kp1[m.queryIdx].pt) 
#             pts2.append(kp2[m.trainIdx].pt) 

#     p1 = np.array(pts1)
#     p2 = np.array(pts2)
#     points = np.array(points)
#     return p1, p2, points

def featureMatching(kp1, kp2, des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des1, des2, k=2)
    points = []
    pts1 = []
    pts2 = []
    for m, n in matches:    
        if m.distance < 0.44 * n.distance:   
            points.append([m])  
            pts1.append(kp1[m.queryIdx].pt) 
            pts2.append(kp2[m.trainIdx].pt) 

    p1 = np.array(pts1)
    p2 = np.array(pts2)
    points = np.array(points)
    return p1, p2, points

def dist3D(dist, pts, K):
    # pts0 = deepcopy(pts[:,0])
    # pts[:,0] = pts[:,1]
    # pts[:,1] = pts0

    Z1D = np.matmul(np.linalg.inv(K), np.concatenate((np.ones((pts.shape[0],1)),pts),axis=1).T)
    coord = np.zeros((pts.shape[0],3))
    for i in range(pts.shape[0]):
        # coord[i,:] = Z1D.T[i,:] / np.linalg.norm(Z1D.T[i,:], ord=2) * dist[int(pts[i, 0]), int(pts[i, 1])]
        coord[i,:] = Z1D.T[i,:] * dist[int(pts[i, 0]), int(pts[i, 1])]
    return coord

def vo(args, left_path, right_path, ldepth_path, rdepth_path, K):
    numfile = len([name for name in os.listdir(left_path)])
    i=10

    # Answer
    # 9.123830746028767091e-01 3.612373103766637339e-01 1.925219282110899321e-01 0.000000000000000000e+00
    # -4.093257579638402976e-01 8.086922246437047157e-01 4.224561833541654599e-01 -1.000000000000000000e+00
    # -3.084009988600033836e-03 -4.642460646884091213e-01 8.857008698344552844e-01 0.000000000000000000e+00


    print("current stage is "+str(i) +" stage")
    imageR= cv.imread(right_path + '//' + str(i) + '_0.png')
    imageL= cv.imread(left_path + '//' + str(i) + '_1.png')
    imageR = cv.cvtColor(imageR, cv.COLOR_BGR2RGB)
    imageL = cv.cvtColor(imageL, cv.COLOR_BGR2RGB)

    depthL = cv.imread(ldepth_path + '//' + str(i) + '_3.pfm',  cv.IMREAD_UNCHANGED)
    depthR = cv.imread(rdepth_path + '//' + str(i) + '_2.pfm',  cv.IMREAD_UNCHANGED)


    _, kpL, desL = orbfeature(imageL)
    _, kpR, desR = orbfeature(imageR)
    pL, pR, matches = featureMatching(kpL, kpR, desL, desR)
    
    # xx = range(1024)
    # yy = range(576)
    # pts_list = []
    # for y in yy:
    #     for x in xx:
    #         pts_list.append([x,y])
    # imagePtsR = np.array(pts_list) 
    # ptsptsR3 = dist3D(depthR.T, imagePtsR, K)

    # ptsptsL3 = dist3D(depthL.T, imagePtsR, K)

    # pc_new = open3d.geometry.PointCloud()
    # pc_new.points = open3d.utility.Vector3dVector(ptsptsR3)
    # open3d.visualization.draw_geometries([pc_new])

    # pc_new.points = open3d.utility.Vector3dVector(ptsptsL3)
    # open3d.visualization.draw_geometries([pc_new])


    ptsL3 = dist3D(depthL.T, pL, K)
    ptsR3 = dist3D(depthR.T, pR, K)

    ptsR3 = np.concatenate((ptsR3, np.ones((ptsR3.shape[0], 1))), axis=1)
    ptsL3 = np.concatenate((ptsL3, np.ones((ptsL3.shape[0], 1))), axis=1)
    n = len(ptsR3)
    n_idx = 50000
    inlier_max = -1
    threshold = 0.05
    inliers_best = []
    for numcheck in range(n_idx):
        inliers=[]
        n_inliers=0
        match_idx = random.sample(range(0,n), 4)
        match1 = np.array([ptsR3[i] for i in match_idx])
        match2 = np.array([ptsL3[i] for i in match_idx])
        if np.linalg.cond(np.matmul(match1.T, match1)) < 1/sys.float_info.epsilon:
            testT = np.matmul(np.matmul(match2.T, match1), np.linalg.inv(np.matmul(match1.T, match1)))
            # testT, _, _ = icp.icp(match1[:,:3], match2[:,:3], max_iterations=20, tolerance=0.1)
            for pts_idx in range(n):
                error = np.linalg.norm(ptsL3[pts_idx].T - np.matmul(testT, ptsR3[pts_idx].T))
                if error<threshold:
                    n_inliers+=1
                    inliers.append(pts_idx)
            if n_inliers>inlier_max:    
                inliers_best = inliers  
                inlier_max = n_inliers  
    print(n)
    print(inlier_max)
    ptsR3 = np.array([ptsR3[i] for i in inliers_best])
    ptsL3 = np.array([ptsL3[i] for i in inliers_best])

    T, _, _ = icp.icp(ptsR3[:,:3], ptsL3[:,:3], max_iterations=5000000, tolerance=0.0001)
    print(T)

    # modelNx1x3 = np.zeros((ptsR3.shape[0], 1, 3), np.float32)
    # modelNx1x3[:, 0, :] = ptsR3[:, :]
    # modelNx1x2 = np.zeros((pL.shape[0], 1, 2), np.float32)
    # modelNx1x2[:, 0, :] = pL[:, :]
    # print(K.shape)
    # print(modelNx1x3.shape)
    # print(modelNx1x2.shape)
    # _, rvec, tvec, _ = cv.solvePnPRansac(modelNx1x3, modelNx1x2, K.astype(np.float32), None, iterationsCount=200000, reprojectionError=0.01)
    # rot, _ = cv.Rodrigues(rvec)

    img3 = cv.drawMatchesKnn(imageL,kpL,imageR,kpR,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='./AI28_20210423_5', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    base_dir = args.base_dir
    tunnel_list = ['tunnel2_fusion']
    left_imgs = '1'
    right_imgs = '0'
    left_depths = '3'
    right_depths = '2'

    # backproj = BackprojectDepth(batch_size=1, height=height_, width=width_)
    # reproject = Project3D(batch_size=1, height=height_, width=width_)

    # fx = 512
    # fy = 288
    # cx = 512
    # cy = 288
    # K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    
    fy = 512
    fz = 288
    cy = 512
    cz = 288
    K = np.array([1, 0, 0, cy, fy, 0, cz, 0, fz]).reshape(3, 3)

    # L/R image path
    for tunnel in tunnel_list:
        base_path = os.path.join(base_dir, tunnel)
        left_path = os.path.join(base_path, left_imgs)
        right_path = os.path.join(base_path, right_imgs)
        ldepth_path = os.path.join(base_path, left_depths)
        rdepth_path = os.path.join(base_path, right_depths)

        # VO
        vo(args, left_path, right_path, ldepth_path, rdepth_path, K)