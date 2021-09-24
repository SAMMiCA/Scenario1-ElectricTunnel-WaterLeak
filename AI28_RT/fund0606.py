import argparse
import os
import numpy as np
import imageio
import cv2 as cv

#Fundamental
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import matplotlib.pyplot as plt

import pdb

def vo(args, left_path, right_path, ldepth_path, rdepth_path, K, f):
    numfile = len([name for name in os.listdir(left_path)])

    i = 2

    print(f[i].split()[:4])
    print(f[i].split()[4:8])
    print(f[i].split()[8:12])
    # Answer
    #1.000000196729367241e+00 - 2.552759737537078071e-07 - 1.584753915669014931e-08 - 1.000000000000000000e+00
    # - 2.174331809886065668e-09    9.646545479889980790e-01  2.635183053615766102e-01 - 1.716613724056514911e-06
    # - 1.884430112777547052e-07 - 2.635184936638867437e-01  9.646546108167074474e-01 - 1.549720764160156250e-06

    print("current stage is " + str(i) + " stage")
    imageR = cv.imread(right_path + '//' + str(i) + '_0.png')
    imageL = cv.imread(left_path + '//' + str(i) + '_1.png')
    imageR = cv.cvtColor(imageR, cv.COLOR_BGR2GRAY)
    imageL = cv.cvtColor(imageL, cv.COLOR_BGR2GRAY)

    w = 75
    h = 75
    imageL = imageL[288-h:288+h, 288-w:288+w]
    imageR = imageR[288-h:288+h, 288-w:288+w]

    depthL = imageio.imread(ldepth_path + '//' + str(i) + '_3.pfm')
    depthR = imageio.imread(rdepth_path + '//' + str(i) + '_2.pfm')

    depthL = depthL[288 - h:288 + h, 288 - w:288 + w]
    depthR = depthR[288 - h:288 + h, 288 - w:288 + w]

    descriptor_extractor = ORB()

    descriptor_extractor.detect_and_extract(imageL)
    keypoints_left = descriptor_extractor.keypoints
    descriptors_left = descriptor_extractor.descriptors


    descriptor_extractor.detect_and_extract(imageR)
    keypoints_right = descriptor_extractor.keypoints
    descriptors_right = descriptor_extractor.descriptors



    matches = match_descriptors(descriptors_left, descriptors_right,
                                cross_check=True)

    # Estimate the epipolar geometry between the left and right image.

    model, inliers = ransac((keypoints_left[matches[:, 0]],
                             keypoints_right[matches[:, 1]]),
                            FundamentalMatrixTransform, min_samples=8,
                            residual_threshold=2, max_trials=10000)

    inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
    inlier_keypoints_right = keypoints_right[matches[inliers, 1]]

    print("Number of matches:", matches.shape[0])
    print("Number of inliers:", inliers.sum())

    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray()

    plot_matches(ax[0], imageL, imageR, keypoints_left, keypoints_right,
                 matches[inliers], only_matches=True)
    plt.show()

    F,_= cv.findFundamentalMat(inlier_keypoints_right,inlier_keypoints_left,cv.FM_LMEDS)
    E = np.matmul(np.matmul(np.transpose(K),F),K)

    R1,R2,T = cv.decomposeEssentialMat(E)

    print(R1)
    print(R2)
    print(T)

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='./AI28_20210423_5', type=str)


if __name__ == '__main__':
    args = parser.parse_args()
    base_dir = args.base_dir
    tunnel_list = ['0603']
    left_imgs = '1'
    right_imgs = '0'
    left_depths = '3'
    right_depths = '2'

    file = open("/mnt/usb0/shyoon/ai28/AI28_RT/AI28_20210423_5/0603/abc.txt").readlines()

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

        # VO
        vo(args, left_path, right_path, ldepth_path, rdepth_path, K, file)