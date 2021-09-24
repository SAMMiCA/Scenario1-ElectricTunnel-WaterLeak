import numpy as np
import os
import torch
import glob
import cv2
import pdb

option= '0609'

image_dir = "./outputTemporal/%s_warp0.2"%option

images_dir = glob.glob(os.path.join(image_dir,"*.png"))

image_len = len(images_dir)

print("image_len:",image_len)
print(images_dir[0])

h,w,_ = cv2.imread(images_dir[0]).shape

print(image_dir+"/"+'%s_video.avi'%option)
out = cv2.VideoWriter(image_dir+"/"+'%s_video.avi'%option,cv2.VideoWriter_fourcc(*"MJPG"), 2, (w,h))

for i in range(image_len):
    img = cv2.imread(os.path.join(image_dir,"%04d.png"%i))
    out.write(img)

out.release()