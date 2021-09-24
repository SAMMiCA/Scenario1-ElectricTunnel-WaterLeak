import os
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


if __name__=='__main__':
    base_dir = '/media/mnt3/AI_project_dataset/210322'
    crack_dir = os.path.join(base_dir, 'crack1_light')
    crack_dir_left = os.path.join(crack_dir, '1')
    crack_dir_right = os.path.join(crack_dir, '0')
    normal_dir = os.path.join(base_dir, 'normal_light_depth_v2')
    normal_dir_left = os.path.join(normal_dir, '1')
    normal_dir_right = os.path.join(normal_dir, '0')
    depth_dir_left = os.path.join(normal_dir, '2')
    depth_dir_right = os.path.join(normal_dir, '3')
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
    K_th = torch.from_numpy(K).float()[None, ...]
    inv_K = K_th.inverse()
    # To tensor
    toTensor = transforms.ToTensor()
    ## hyper-parameter ##
    height_ = 576
    width_ = 1024
    backproj = BackprojectDepth(batch_size=1, height=height_, width=width_)
    reproject = Project3D(batch_size=1, height=height_, width=width_)
    for i in range(len_of_img):
        ####
        zeros = torch.zeros((3,1))
        cur_trans = trans_mat[i].split(' ')
        cur_trans_mat = torch.zeros((12, 1))
        for j in range(12):
            cur_trans_mat[j] = float(cur_trans[j])
        cur_trans_mat = cur_trans_mat.reshape(3, 4).float()[None, ...]
        cur_trans_mat[0, :, 3] = torch.Tensor([-1, 0, 0])
        ####
        img_name_format_left = str(i) + '_1.png'
        img_name_format_right = str(i) + '_0.png'
        depth_name_left = str(i) +'_2.pfm'
        depth_name_right = str(i) +'_3.pfm'
        cur_left_crack_filename = os.path.join(crack_dir_left, img_name_format_left)
        cur_right_normal_filename = os.path.join(normal_dir_right, img_name_format_right)
        cur_left_depth_filename = os.path.join(depth_dir_left, depth_name_left)
        cur_right_depth_filename = os.path.join(depth_dir_right, depth_name_right)
        cur_left_crack = cv2.imread(cur_left_crack_filename)
        cur_right_normal = cv2.imread(cur_right_normal_filename)
        cur_left_depth = imageio.imread(cur_left_depth_filename)
        cur_right_depth = imageio.imread(cur_right_depth_filename)
        cur_right_depth_th = toTensor(cur_right_depth)
        cam_pts = backproj(cur_right_depth_th, inv_K)
        pixel_coords = reproject(cam_pts, K_th, cur_trans_mat)

        cur_right_normal_th = toTensor(cur_right_normal)[None, ...].float()
        cur_left_restored = F.grid_sample(cur_right_normal_th, pixel_coords, padding_mode='zeros', align_corners=True)
        cur_left_restored = (255*cur_left_restored.squeeze().cpu().numpy().transpose(1,2,0)).astype(np.uint8)