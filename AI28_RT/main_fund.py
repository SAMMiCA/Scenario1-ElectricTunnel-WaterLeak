import numpy as np
import cv2 as cv
import os
import imageio
import sys


import feature_extraction
import coordinate
import icp_ocg
import RANSAC
import random
import argparse

from scipy.spatial.transform import Rotation as rot


import torch


def vo(args, left_path, right_path, ldepth_path, rdepth_path, K, f,idx):
    numfile = len([name for name in os.listdir(left_path)])
    i = idx

    # print(f[i].split()[:4])
    # print(f[i].split()[4:8])
    # print(f[i].split()[8:12])

    temp= f[i].split()

    gt_R = np.array([[float(temp[0]),float(temp[1]),float(temp[2])],
                    [float(temp[4]),float(temp[5]),float(temp[6])],
                    [float(temp[8]), float(temp[9]), float(temp[10])]])

    # print(gt_R)
    # x_rot =

    # gt_R = gt_R*

    gt_R = rot.from_matrix(gt_R)
    gt_R_euler = gt_R.as_euler('xyz',degrees=True)


    # Answer
    # 9.123830746028767091e-01 3.612373103766637339e-01 1.925219282110899321e-01 0.000000000000000000e+00
    # -4.093257579638402976e-01 8.086922246437047157e-01 4.224561833541654599e-01 -1.000000000000000000e+00
    # -3.084009988600033836e-03 -4.642460646884091213e-01 8.857008698344552844e-01 0.000000000000000000e+00

    print("current stage is " + str(i) + " stage")
    imageR = cv.imread(right_path + '//' + str(i) + '_0.png')
    imageL = cv.imread(left_path + '//' + str(i) + '_1.png')
    imageR = cv.cvtColor(imageR, cv.COLOR_BGR2RGB)
    imageL = cv.cvtColor(imageL, cv.COLOR_BGR2RGB)

    # w = 200
    # h = 150

    # imageL = imageL[288 - h:288 + h, 288 - w:288 + w]
    # imageR = imageR[288 - h:288 + h, 288 - w:288 + w]

    depthL = imageio.imread(ldepth_path + '//' + str(i) + '_3.pfm')
    depthR = imageio.imread(rdepth_path + '//' + str(i) + '_2.pfm')

    # depthL = depthL[288 - h:288 + h, 288 - w:288 + w]
    # depthR = depthR[288 - h:288 + h, 288 - w:288 + w]

    keypoint1, keypoint2 = feature_extraction.feature(imageL,imageR)
    # pdb.set_trace()
    print("Before RANSAC", len(keypoint1), len(keypoint2))

    p1, p2 = coordinate.coordinate(keypoint1,keypoint2,depthL,depthR,K)
    p1, p2 = RANSAC.ransac(p1,p2)

    print("RANSAC",len(p1),len(p2))

    RT_mat = icp_ocg.icp_func(p1,p2)

    R = RT_mat[:3,:3]
    # print(R)

    r = rot.from_matrix(R)
    R_euler = r.as_euler('xyz',degrees=True)
    print("GT Euler", gt_R_euler)
    print("Pred Euler",R_euler)

    # R_inv = np.linalg.inv(R)

    # print(RT_mat)
    # print(R_inv)




parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='./AI28_20210423_5', type=str)
parser.add_argument('--index',required=True,type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    base_dir = args.base_dir
    tunnel_list = ['0603']
    left_imgs = '1'
    right_imgs = '0'
    left_depths = '3'
    right_depths = '2'

    # backproj = BackprojectDepth(batch_size=1, height=height_, width=width_)
    # reproject = Project3D(batch_size=1, height=height_, width=width_)

    fx = 288
    fy = 288
    cx = 288
    cy = 288
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)

    # fy = 512
    # fz = 288
    # cy = 512
    # cz = 288
    # K = np.array([1, 0, 0, cy, fy, 0, cz, 0, fz]).reshape(3, 3)

    # L/R image path
    for tunnel in tunnel_list:
        base_path = os.path.join(base_dir, tunnel)
        left_path = os.path.join(base_path, left_imgs)
        right_path = os.path.join(base_path, right_imgs)
        ldepth_path = os.path.join(base_path, left_depths)
        rdepth_path = os.path.join(base_path, right_depths)

        file = open("/mnt/usb0/shyoon/ai28/AI28_RT/AI28_20210423_5/0603/abc.txt").readlines()

        # VO
        vo(args, left_path, right_path, ldepth_path, rdepth_path, K, file, args.index)