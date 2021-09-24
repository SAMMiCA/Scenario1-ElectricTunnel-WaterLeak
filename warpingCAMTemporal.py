import os
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
import skimage

######CAM
import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2

import torch.nn.functional as F
# from torchvision.models.resnet import resnet101, resnet50
from PIL import Image
from network.resnet import resnet50

import pdb
import matplotlib.pyplot as plt

###Fundamental Matrix Estimation###
from scipy.spatial.transform import Rotation as scirot

from AI28_RT import feature_extraction
from AI28_RT import coordinate
from AI28_RT import icp_ocg
from AI28_RT import RANSAC
from AI28_RT import icp_ocg


def returnCAM(feature_conv, weight_softmax, img, h_x,labels):
    # im_h, im_w = img.shape[1], img.shape[0]
    im_h, im_w = img.size()[2],img.size()[3]
    # print(im_h,im_w)
    probs, idx = h_x.sort(0, True)


    size_upsample = (im_w, im_h)  #####BE CAREFUL
    batch, nc, h, w = feature_conv.shape
    output_cam = []

    cams = torch.zeros((batch, 2, h,w))
    # fgs = torch.zeros((batch,1,im_h,im_w))

    for i in range(batch):


        cam_dict = {}

        for j in range(2):
            # j = j.item()
            cam = torch.mm(weight_softmax[j].clone().unsqueeze(0), (feature_conv[i].reshape((nc, h * w))))
            cam = cam.reshape(h, w)
            cam = cam - torch.min(cam)
            cam = cam / torch.max(cam)

            cams[i, j, :, :] = cam#*labels[j]
            cam_dict[j] =  cv2.resize(cam.cpu().detach().numpy(),size_upsample)

    cams = F.interpolate(cams,(im_h,im_w),mode='bilinear',align_corners=False)

    return cams
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

def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

def vo(left_path, right_path, ldepth_path, rdepth_path, K):

    imageR = cv2.imread(right_path + '//' + str(i-1) + '_1.png')
    imageL = cv2.imread(left_path + '//' + str(i) + '_1.png')

    imageR = cv2.cvtColor(imageR, cv2.COLOR_BGR2RGB)
    imageL = cv2.cvtColor(imageL, cv2.COLOR_BGR2RGB)

    depthL = imageio.imread(ldepth_path + '//' + str(i) + '_3.pfm')
    depthR = imageio.imread(rdepth_path + '//' + str(i-1) + '_3.pfm')


    keypoint1, keypoint2 = feature_extraction.feature(imageL,imageR)

    # print("Before RANSAC", len(keypoint1), len(keypoint2))

    p1, p2 = coordinate.coordinate(keypoint1,keypoint2,depthL,depthR,K)
    p1, p2 = RANSAC.ransac(p1,p2)

    # print("RANSAC",len(p1),len(p2))

    RT_mat = icp_ocg.icp_func(p1,p2)

    R = RT_mat[:3,:3]

    r = scirot.from_matrix(R)
    R_euler = r.as_euler('xyz',degrees=True)
    print("GT Euler", gt_R_euler[0],gt_R_euler[1],gt_R_euler[2])
    print("Pred Euler", R_euler)
    # print("Pred Euler", R_euler[1],R_euler[2],R_euler[0])

    return  R

if __name__=='__main__':
    base_dir = '/mnt/usb0/shyoon/ai28/AI28_RT/AI28_20210423_5'
    option= '0609' #crack1/2/3, frost 1/2/3
    option2= 'light' #tunnel2

    crack_dir = os.path.join(base_dir, '%s'%(option))
    crack_dir_left = os.path.join(crack_dir, '1') #1_crack
    # crack_dir_right = os.path.join(crack_dir, '0')
    normal_dir = os.path.join(base_dir, '%s'%option)
    normal_dir_left = os.path.join(normal_dir, '1')
    depth_dir_left = os.path.join(normal_dir, '3')
    img_list_normal_left = os.listdir(normal_dir_left)
    len_of_img = len(img_list_normal_left)
    # transformation matrix
    crack_txt_filename = os.path.join(normal_dir, 'abc.txt')
    f1 = open(crack_txt_filename)
    f2 = open(crack_txt_filename)

    trans_mat = f1.read().split('\n')
    f_temp = f2.readlines()
    # intrinsic parameter
    fx = 288
    fy = 288
    cx = 288
    cy = 288
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    # K = np.array([1, 0, 0, cy, fy, 0, cx, 0, fx]).reshape(3, 3)
    K_th = torch.from_numpy(K).float()[None, ...]
    inv_K = K_th.inverse()
    # To tensor
    toTensor = transforms.ToTensor()
    ## hyper-parameter ##
    height_ = 576
    width_ = 576
    backproj = BackprojectDepth(batch_size=1, height=height_, width=width_)
    reproject = Project3D(batch_size=1, height=height_, width=width_)
    ##CAM
    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]

    inv_normalize = transforms.Normalize(
        mean=[-mean_img[0] / std_img[0], -mean_img[1] / std_img[1], -mean_img[2] / std_img[2]],
        std=[1 / std_img[0], 1 / std_img[1], 1 / std_img[2]]
    )
    resnet = resnet50(pretrained=True).cuda()

    state_dict = torch.load("./model/model_resnet_er_v3.pth")
    resnet.load_state_dict(state_dict)
    resnet.eval()



    # To tensor
    toTensor = transforms.ToTensor()

    #Left abnormal
    #Right normal
    cnt = 0

    for i in range(9, len_of_img):
        temp = f_temp[i].split()

        gt_R = np.array([[float(temp[0]), float(temp[1]), float(temp[2])],
                         [float(temp[4]), float(temp[5]), float(temp[6])],
                         [float(temp[8]), float(temp[9]), float(temp[10])]])

        gt_R = scirot.from_matrix(gt_R)
        gt_R_euler = gt_R.as_euler('xyz', degrees=True)

        img_name_format_left = str(i) + '_1.png'

        depth_name_left = str(i) + '_3.pfm'
        img_name_format_left_past = str(i-1) + '_1.png'
        depth_name_left_past = str(i-1) + '_3.pfm'

        cur_left_crack_filename = os.path.join(crack_dir_left, img_name_format_left)
        cur_left_depth_filename = os.path.join(depth_dir_left, depth_name_left)
        cur_left_normal_filename = os.path.join(normal_dir_left, img_name_format_left)

        cur_left_crack = cv2.imread(cur_left_crack_filename)
        cur_left_normal = cv2.imread(cur_left_normal_filename)
        cur_left_normal_gray = cv2.imread(cur_left_normal_filename,flags=cv2.IMREAD_GRAYSCALE)
        cur_left_depth = imageio.imread(cur_left_depth_filename)

        if i>=1:
            past_left_normal_filename = os.path.join(normal_dir_left, img_name_format_left_past)
            past_left_depth_filename = os.path.join(depth_dir_left, depth_name_left_past)
            past_left_normal = cv2.imread(past_left_normal_filename)
            past_left_depth = imageio.imread(past_left_depth_filename)
            past_left_depth_th = toTensor(past_left_depth) #right_depth

        # For pose estimation network
        # print(future_left_normal_filename)

        rotation = vo(normal_dir_left, normal_dir_left,
                      depth_dir_left, depth_dir_left, K)

        rotation = torch.from_numpy(rotation)

        # For Network / input crack left
        image = Image.open(cur_left_crack_filename).convert("RGB")
        image = np.float64(image) / 255.0
        image = image - np.array([0.485, 0.456, 0.406])
        image = image / np.array([0.229, 0.224, 0.225])
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).type(torch.FloatTensor)

        _,_,H,W = image.size()

        with torch.no_grad():
            output, features = resnet(image.cuda())
        pred = output[0].cpu().detach().numpy()
        pred_cls = pred.argmax() #pred_cls =0 : abnormal pred_cls =1 normal

        feature4 = features[3]

        h_x = F.softmax(output, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        params = list(resnet.parameters())
        weight_softmax = params[-2]

        CAMs = returnCAM(feature4, weight_softmax, image, h_x, pred)
        CAMs = CAMs.detach().cpu().numpy()
        CAMs_crack = CAMs[0][idx[0]]

        # rot_pred = scirot.from_matrix(rotation)
        # R_euler = rot_pred.as_euler('xyz', degrees=True)
        # print("Pred Euler", R_euler)

########################################
        import transforms3d.euler as t3d
        # #
        # cur_trans = trans_mat[i].split(' ')
        # cur_trans_mat = np.zeros((12, 1))
        # for j in range(12):
        #     cur_trans_mat[j] = float(cur_trans[j])
        #
        # cur_trans_mat = cur_trans_mat.reshape(3, 4)
        #
        # cur_trans_mat = torch.from_numpy(cur_trans_mat).float()[None, ...]
        # # cur_trans_mat[:, :3, :3] = cur_trans_mat[:, :3, :3]
        # np_rot = np.asarray(cur_trans_mat[:, :3, :3].squeeze())
        # roll, pitch, yaw = t3d.mat2euler(np_rot)
        # roll_new = roll
        # pitch_new = pitch
        # yaw_new = yaw
        # # print(roll*180/np.pi,pitch*180/np.pi,yaw*180/np.pi)
        #
        # new_trans_mat = t3d.euler2mat(roll_new, pitch_new, yaw_new, 'sxyz')
        # # rot = torch.from_numpy(np.linalg.inv(new_trans_mat))
        # rot = torch.from_numpy(new_trans_mat)
        # cur_trans_mat[:, :3, :3] = rot
        # cur_trans_mat[0, :, 3] = torch.Tensor([-0.5, 0.0, 0])

        ######################################
        #
        cur_trans_mat = torch.zeros((3, 4))

        # cur_trans_mat[:3,:3]=rotation
        rot_pred = scirot.from_matrix(rotation)
        R_euler = rot_pred.as_euler('xyz', degrees=True)

        roll_new = -np.array(R_euler[0])*np.pi/180
        pitch_new = -np.array(R_euler[1])*np.pi/180
        yaw_new = -np.array(R_euler[2])*np.pi/180

        # pdb.set_trace()
        new_trans_mat = t3d.euler2mat(roll_new, pitch_new, yaw_new, 'sxyz')
        rot = torch.from_numpy(np.linalg.inv(new_trans_mat))
        # rot = torch.from_numpy(new_trans_mat)
        cur_trans_mat = cur_trans_mat.unsqueeze(0)
        cur_trans_mat[:, :3, :3] = rot

        # new_trans = rot * torch.Tensor([-1.0, 0, 0])
        # cur_trans_mat[0, :, 3] = new_trans[:, 0]
        # cur_trans_mat[0, :, 3] = torch.Tensor([-0.5, 0.0, 0])
        cur_trans_mat[0, :, 3] = torch.Tensor([0, 0.0, -0.2])

##################################################

        cam_pts = backproj(past_left_depth_th, inv_K)
        pixel_coords = reproject(cam_pts, K_th, cur_trans_mat)

        if i==9:
            warped = past_left_normal

        # past_left_normal_th = toTensor(past_left_normal)[None, ...].float()
        past_left_normal_th = toTensor(warped)[None, ...].float()
        cur_left_restored = F.grid_sample(past_left_normal_th, pixel_coords, padding_mode='zeros', align_corners=True)
        # cur_left_restored = (255 * cur_left_restored.squeeze().cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)

        left_restored = cur_left_restored[0].permute(1, 2, 0).detach().cpu().numpy()
        left_ori = cur_left_crack
        cl = CAMs_crack < 0.4
        ncl = CAMs_crack >= 0.4
        cl_stack = np.stack((cl, cl, cl), 2)
        ncl_stack = np.stack((ncl, ncl, ncl), 2)

        # pdb.set_trace()
        if pred_cls==0:
            left_final = (255 * left_restored * (1 - cl_stack) + left_ori * cl_stack) / 255
            error = np.abs(cv2.cvtColor(np.uint8(255 * left_final), cv2.COLOR_BGR2GRAY) - cv2.cvtColor(cur_left_normal,cv2.COLOR_BGR2GRAY))
        else:
            left_final = left_ori
            error = np.zeros((H,W))

        warped = left_final
        # warped = cv2.cvtColor(warped,cv2.COLOR_RGB2BGR)
        # past_left_normal = left_final
        # past_left_depth = cur_left_depth
        # past_left_depth_th = toTensor(past_left_depth) #right_depth

        fig = plt.figure(figsize=(16, 9), dpi=120)
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3, 2, 4)
        ax5 = fig.add_subplot(3, 2, 5)
        ax6 = fig.add_subplot(3, 2, 6)

        ax1.title.set_text('%s image(left)'%option)
        ax2.title.set_text('%s image(left) with CAM'%option)
        ax3.title.set_text('normal image(past left)')
        ax4.title.set_text('warped normal image(past2current)')
        ax5.title.set_text('reconstructed image(left)')
        ax6.title.set_text('normal image(left)')
        # ax6.title.set_text('error map')

        ax1.imshow(left_ori)
        ax2.imshow(left_ori)
        if pred_cls == 0:
            ax2.imshow(np.uint8(255 * CAMs[0][idx[0]]), cmap='jet', alpha=0.3)
        ax3.imshow(past_left_normal)
        ax4.imshow(left_restored)
        ax5.imshow(left_final)
        ax6.imshow(cur_left_normal)
        # ax6.imshow(error)


        # plt.show()
        plt.savefig(os.path.join('./outputTemporal','%s_warp0.2'%option,'%04d.png'%cnt))
        plt.close()
        print(os.path.join('./outputTemporal','%s_warp'%option,'%04d.png'%cnt))
        print("saving figure %04d"%cnt)
        cnt+=1
        print("="*60)

