import cv2
import numpy as np
from AI28_RT.icp import best_fit_transform
import math

# 오차를 계산합니다

def dist(list1,list2):
    list_len = len(list1)
    dist_db = 0
    loss = []
    for num in range(list_len):
        for i in range(len(list1[num])):
            dist_db += abs(list1[num][i]**2-list2[num][i]**2)
        loss.append(math.sqrt(dist_db))
        dist_db = 0
    return loss
# obtain
def obtain_T(keypoint1,keypoint2):

    point_num = len(keypoint1)

    random = []


    while not len(random) == 4:
        random_num = np.random.randint(0,point_num)

        if random_num not in random:
            random.append(random_num)

    point1 = []
    point2 = []
    key11 = []
    key22 = []

    for i in random:
        key1 = keypoint1[i]+[1]
        key11.append(key1)
        key2 = keypoint2[i]+[1]
        key22.append(key2)

        point1.append(keypoint1[i])
        point2.append(keypoint2[i])

    point1_array = np.array(point1)
    point2_array = np.array(point2)

    T,R,t = best_fit_transform(point1_array,point2_array)

    return T

def inlier(keypoint1,keypoint2):

    T = obtain_T(keypoint1,keypoint2)

    point_num = len(keypoint1)

    hom_key11 = []
    hom_key22 = []

    for j in range(point_num):
        hom_key1 = keypoint1[j]+[1]
        hom_key11.append(hom_key1)
        hom_key2 = keypoint2[j]+[1]
        hom_key22.append(hom_key2)

    key1_hom_array = np.array(hom_key11)

    pre_point2 = np.transpose(np.matmul(T, np.transpose(key1_hom_array)))
    pre_point2_list = pre_point2.tolist()
    loss = dist(pre_point2_list, hom_key22)

    inlier_num = 0

    pts1 = []
    pts2 = []
    # print(point_num)
    inlier_idx = 0

    for i in loss:

        if i < 2.0:
            inlier_num += 1
            pts1.append(keypoint1[inlier_idx])
            pts2.append(keypoint2[inlier_idx])
        inlier_idx += 1

    return inlier_num,T,pts1,pts2


def ransac(pt1,pt2):

    pre_num = 0
    real_t = 0

    for i in range(3000):
        num, T, pts1, pts2 = inlier(pt1,pt2)
        if num > pre_num:
            real_t = T
            pre_num = num
            inlier_pts1 = pts1
            inlier_pts2 = pts2

    return inlier_pts1,inlier_pts2

