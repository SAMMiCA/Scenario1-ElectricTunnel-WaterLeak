import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pdb

dataset_path = "C:\\Users\\sunghoon Yoon\\PycharmProjects\\AI28\\train.txt"
img_gt_name_list = open(dataset_path).read().splitlines()
img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

cls_labels_dict = np.load('./cls_labels.npy',allow_pickle=True).item()
# label_list= [cls_labels_dict[img_name] for img_name in img_name_list]

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']


AE_dir = "D:\\ICCV2021\\figures\\1_introduction\\all_erase\\train"

fuse_dir = "D:\\ICCV2021\\figures\\1_introduction\\erase_infer\\val"

base_dir = "D:\\ICCV2021\\figures\\1_introduction\\baseline\\infer"

our_dir = "D:\\ICCV2021\\figures\\1_introduction\\ours\\train"

save_dir = "D:\\ICCV2021\\figures\\1_introduction\\output"

for img_name in img_name_list:

    labels = cls_labels_dict[img_name]
    label_list =[]

    # pdb.set_trace()
    for i in range(20):

        if labels[i] == 1 :
            label_list.append(CAT_LIST[i])

    # pdb.set_trace()
    # print(label_list)

    for label in label_list:

        # pdb.set_trace()

        AE= cv2.imread(AE_dir+"\\042_%s_cam_%s.png"%(img_name,label))
        AE = cv2.cvtColor(AE,cv2.COLOR_BGR2RGB)
        fuse = cv2.imread(fuse_dir+"\\000_%s_cam_%s_fusion.png"%(img_name,label))
        fuse = cv2.cvtColor(fuse, cv2.COLOR_BGR2RGB)
        base = cv2.imread(base_dir+"\\042_%s_cam_%s.png"%(img_name,label))
        base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        our = cv2.imread(our_dir+"\\000_%s_cam_%s.png"%(img_name,label))
        our = cv2.cvtColor(our, cv2.COLOR_BGR2RGB)

        fig,ax = plt.subplots(1,4,figsize=(15,7))

        ax[0].imshow(base)
        ax[0].set_title('baseline')
        ax[1].imshow(fuse)
        ax[1].set_title('iterative')
        ax[2].imshow(AE)
        ax[2].set_title('all erasing')
        ax[3].imshow(our)
        ax[3].set_title('Ours')

        fig.suptitle('%s'%img_name)
        plt.show()


