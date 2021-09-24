import os
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
import skimage


def pixel2cam(depth, pixel_coord, fx, fy, cx, cy):
    x = (pixel_coord[0, :] - cx) / fx * pixel_coord[2, :]
    y = (pixel_coord[1, :] - cy) / fy * pixel_coord[2, :]
    z = pixel_coord[2, :]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    output = depth.reshape(-1, 1)*cam_coord
    return output

def cam2pixel(pix2cam, K, rot, trans):
    pcoords = np.matmul(rot, pix2cam.transpose())
    pcoords = pcoords + trans[:, None]
    pcoords = np.matmul(K, pcoords)
    X = pcoords[0, :]
    Y = pcoords[1, :]
    Z = pcoords[2, :]
    X_norm = 2*(X/Z)/(width - 1) - 1
    Y_norm = 2*(Y/Z)/(height - 1) - 1
    X_norm = X_norm.reshape(height, width)
    Y_norm = Y_norm.reshape(height, width)
    pixel_coords = np.stack((X_norm, Y_norm), axis=2)
    return pixel_coords

def set_id_grid(depth):
    global height
    global width
    height, width = depth.shape
    j_range = np.repeat(np.arange(0, height), width).reshape(height, width)
    i_range = np.repeat(np.arange(0, width), height).reshape(width, height).transpose()
    ones = np.ones((height, width))
    pixel_coords = np.stack((i_range, j_range, ones), axis=0)
    pixel_coords = pixel_coords.reshape(3, -1)
    return pixel_coords

if __name__=='__main__':
    base_dir = '/mnt/usb0/shyoon/ai28/dataset'
    crack_dir = os.path.join(base_dir, 'crack1_tunnel2')
    crack_dir_left = os.path.join(crack_dir, '1')
    crack_dir_right = os.path.join(crack_dir, '0')
    normal_dir = os.path.join(base_dir, 'normal_tunnel2')
    normal_dir_left = os.path.join(normal_dir, '1')
    normal_dir_right = os.path.join(normal_dir, '0')
    depth_dir_left = os.path.join(normal_dir, '3')
    depth_dir_right = os.path.join(normal_dir, '4')
    img_list_normal_left = os.listdir(normal_dir_left)
    len_of_img = len(img_list_normal_left)
    # transformation matrix
    crack_txt_filename = os.path.join(crack_dir, 'abc.txt')
    f = open(crack_txt_filename, 'r')
    trans_mat = f.read().split('\n')
    # intrinsic parameter
    fx = 512
    fy = 512
    cx = 512
    cy = 288
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    # To tensor
    toTensor = transforms.ToTensor()

    #Left abnormal
    #Right normal

    for i in range(len_of_img):
        img_name_format_left = str(i) + '_1.png'
        img_name_format_right = str(i) + '_0.png'
        # depth_name = str(i) +'_3.pfm'
        depth_name = '%d_3.pfm'%i
        cur_left_crack_filename = os.path.join(crack_dir_left, img_name_format_left)
        cur_left_depth_filename = os.path.join(depth_dir_left, depth_name)
        cur_right_normal_filename = os.path.join(normal_dir_right, img_name_format_right)
        cur_left_crack = cv2.imread(cur_left_crack_filename)
        cur_right_normal = cv2.imread(cur_right_normal_filename)
        # print(cur_left_depth_filename)
        cur_left_depth = imageio.imread(cur_left_depth_filename) #If Error: script: imageio_download_bin freeimage
        # cur_left_depth = skimage.io.imread(cur_left_depth_filename, plugin='matplotlib')
        pixel_coords = set_id_grid(cur_left_depth)
        pix2cam = pixel2cam(cur_left_depth, pixel_coords, fx, fy, cx, cy) #Camera coordinate (left crack)
        cur_trans = trans_mat[i].split(' ')
        cur_trans_mat = np.zeros((12, 1))
        for j in range(12):
            cur_trans_mat[j] = float(cur_trans[j])
        cur_trans_mat = cur_trans_mat.reshape(3, 4)
        cur_rot_mat = cur_trans_mat[:3, :3]
        cur_translation = cur_trans_mat[:3, 3]
        cur_translation = np.array([-1, 0, 0])
        pixel_coords = cam2pixel(pix2cam, K, cur_rot_mat, cur_translation) #warping in cam coord, and camcoord2pixcoord (left crack to right crack (warped))
        pixel_coords_th = torch.from_numpy(pixel_coords)[None, ...].float()
        cur_right_normal_th = toTensor(cur_right_normal)[None, ...].float()
        cur_left_restored = F.grid_sample(cur_right_normal_th, pixel_coords_th, padding_mode='zeros', align_corners=True)
        plt.imshow(cur_left_restored[0].permute(1,2,0).detach().cpu().numpy())
        plt.show()