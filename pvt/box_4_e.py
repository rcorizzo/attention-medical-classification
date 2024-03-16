#!/usr/bin/env python
# coding: utf-8

# In[15]:


import cv2
from matplotlib import pyplot as plt
import os
from glob import glob
import numpy as np


# In[ ]:


def box_define(train_path,dimension=224):
    paths = glob(os.path.join(train_path, '*','*.png')) 

    names=[]
    for path in paths:
        name=os.path.basename(path)
        names.append(name)

    # Extract corresponding masks
    mask_path='../data/Ottawa_masks_512'
    masks_dir=[] 
    for img_name in names:
        mask_dir=os.path.join(mask_path, img_name)
        masks_dir.append(mask_dir)
    
    # If no corresponding masks, take all masks as reference to calculate
    if not os.path.exists(masks_dir[0]):
        masks_dir=glob(os.path.join(mask_path, '*.png')) 

    e_x,e_y,e_w,e_h=[],[],[],[]
    for mask in masks_dir:
        img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (dimension, dimension))
        img_h, img_w = img.shape

        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        xs,ys,ws,hs=[],[],[],[]

        for contour in contours:
            X,Y,W,H=cv2.boundingRect(contour)
            xs.append(X)
            ys.append(Y)
            ws.append(X+W)
            hs.append(Y+H)

        contour_x=min(xs)
        contour_y=min(ys)
        contour_w=max(ws)-contour_x
        contour_h=max(hs)-contour_y

        box_x=contour_x
        box_y=contour_y+contour_h/2
        box_h=contour_h/2
        box_w=contour_w

        e_x.append(box_x)
        e_y.append(box_y)
        e_w.append(box_w)
        e_h.append(box_h)
        
    avg_x=round(np.mean(e_x))
    avg_y=round(np.mean(e_y))
    avg_h=round(np.mean(e_h))
    avg_w=round(np.mean(e_w))
    
    return avg_x,avg_y,avg_w,avg_h

