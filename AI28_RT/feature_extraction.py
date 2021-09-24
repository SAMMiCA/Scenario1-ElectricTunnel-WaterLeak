import cv2
import matplotlib.pyplot as plt
import numpy as np




def feature(impath,impath1):

    img1 = impath#cv2.imread(impath,0)
    img2 = impath1#cv2.imread(impath1,0)


    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)


    bf = cv2.BFMatcher()


    matches = bf.knnMatch(des1,des2, k=2)

    good = []

    key_point1 = []
    key_point2 = []


    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])
            key_point1.append(kp1[m.queryIdx].pt)
            key_point2.append(kp2[m.trainIdx].pt)

    return key_point1,key_point2