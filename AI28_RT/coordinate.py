import cv2
import numpy as np
import math


def coordinate(key_point1,key_point2,depth1,depth2, K):

    # img1 = cv2.imread(d_impath,cv2.IMREAD_UNCHANGED)
    # img2 = cv2.imread(d_impath1,cv2.IMREAD_UNCHANGED)
    img1 = depth1
    img2 = depth2

    num_points = len(key_point1)

    # print(K)
    K_inv = np.linalg.inv(K)
    # print(K_inv)

    point1 = []
    point2 = []

    # dis = stereo(key_point1,key_point2)

    for point in range(num_points):
        x = key_point1[point][0]
        y = key_point1[point][1]
        # image_coord.append([x,y,1])

        distance = img1[int(y), int(x)] #/655.35

        image_coord = np.matmul(K_inv, [[x], [y], [1]])

        X = image_coord[0]
        Y = image_coord[1]


        point1.append([(distance*X).tolist()[0],(distance * Y).tolist()[0],distance])

    for point in range(num_points):
        x = key_point2[point][0]
        y = key_point2[point][1]
        # image_coord.append([x,y,1])

        distance = img2[int(y), int(x)] #/655.35

        # distance = dis[point]

        image_coord = np.matmul(K_inv, [[x], [y], [1]])

        X = image_coord[0]
        Y = image_coord[1]

        point2.append([(distance * X ).tolist()[0],(distance * Y ).tolist()[0],distance])

    return point1, point2
# print(len(distance))
# print(len(distance1))
# print(point1)
# plt.imshow(img1),plt.show()

