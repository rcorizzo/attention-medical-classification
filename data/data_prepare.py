#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os, sys
from os import listdir
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from glob import glob
from sklearn.model_selection import KFold
import pathlib

import torch
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
from torchvision.models import vgg16, VGG16_Weights
import pathlib
import timm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import pickle


# In[ ]:


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# In[ ]:


df=pd.read_csv('NIH/Data_Entry_2017_v2020.csv')

def class_names(df,target):
    lst=[]
    for i in range(df.shape[0]):
        label=df['Finding Labels'][i]
        if label not in target:
            continue
        elif label==target:
            lst.append(df['Image Index'][i])
    return lst


# In[ ]:


# Function to check intersection of two lists
def check_inter(lst1,lst2):
    extracted1=[item.split('_')[0] for item in lst1]
    set1=set(extracted1)

    extracted2=[item.split('_')[0] for item in lst2]
    set2=set(extracted2)

    intersection = set1 & set2
    
    return intersection


# In[ ]:


data_path='NIH'
images = sorted(glob(os.path.join(data_path, "*", "*.png")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))


# In[ ]:


def get_path(images,lst):
    paths=[]
    for path in tqdm(images):
        if path.split('/')[-1] not in lst:
            continue
        else:
            paths.append(path)
    return paths


# ### Prepare data for multi-class classification: Effusion, Pneumothorax and No Finding. 
# Three subsets: training, validation and testing

# In[ ]:


e=class_names(df,'Effusion')
p=class_names(df,'Pneumothorax')
nf=class_names(df,'No Finding')
print(len(e),len(p),len(nf))

n_train = round(len(p)*0.6)
n_val=round(len(p)*0.2)

p_train=p[0:n_train]
p_val=p[n_train:n_train+n_val]
p_test=p[n_train+n_val:n_train+n_val*2]

print(len(p_train),len(p_val),len(p_test))


# In[ ]:


# Exchange overlap patients' id
index1 = p_train.index('00017801_004.png')
index2 = p_val.index('00017942_002.png')

p_train[index1], p_val[index2] = p_val[index2], p_train[index1]

index1 = p_val.index('00024749_000.png')
index2 = p_test.index('00025082_007.png')

p_val[index1], p_test[index2] = p_test[index2], p_val[index1]


# In[ ]:


e_train=[]
for item in e:
    if item.split('_')[0] in [i.split('_')[0] for i in p_val]:
        continue
    elif item.split('_')[0] in [i.split('_')[0] for i in p_test]:
        continue
    else:
        e_train.append(item)

random.seed(42)
e_train = random.sample(e_train, len(p_train))

e_val=[]
for item in e:
    if item.split('_')[0] in [i.split('_')[0] for i in p_train]:
        continue
    elif item.split('_')[0] in [i.split('_')[0] for i in p_test]:
        continue
    elif item.split('_')[0] in [i.split('_')[0] for i in e_train]:
        continue
    else:
        e_val.append(item)

random.seed(42)
e_val = random.sample(e_val, len(p_val))

e_test=[]
for item in e:
    if item.split('_')[0] in [i.split('_')[0] for i in p_train]:
        continue
    elif item.split('_')[0] in [i.split('_')[0] for i in e_train]:
        continue
    elif item.split('_')[0] in [i.split('_')[0] for i in p_val]:
        continue
    elif item.split('_')[0] in [i.split('_')[0] for i in e_val]:
        continue
    else:
        e_test.append(item)
        
random.seed(42)
e_test = random.sample(e_test, len(p_test))


# In[ ]:


ep=e+p
extract_ep=[item.split('_')[0] for item in ep]
uni_ep=set(extract_ep)

extract_nf=[item.split('_')[0] for item in nf]
uni_nf=set(extract_nf)

difference = [item for item in uni_nf if item not in uni_ep]  #Filter out patient ids that are in nf but not in e&p
print(len(difference))


# In[ ]:


# Split remained no find images to three parts and make there are no same patients
random.seed(42)
sublist1 = random.sample(difference, len(p_train))
remaining_list = [item for item in difference if item not in sublist1]
sublist2 = random.sample(remaining_list, len(p_val))
remaining_list = [item for item in remaining_list if item not in sublist2]
sublist3 = random.sample(remaining_list, len(p_test))

nf_train,nf_val,nf_test=[],[],[]
for name in nf:
    if name.split('_')[0] in sublist1:
        nf_train.append(name)
    elif name.split('_')[0] in sublist2:
        nf_val.append(name)
    elif name.split('_')[0] in sublist3:
        nf_test.append(name)

print(len(nf_train),len(nf_val),len(nf_val))


# In[ ]:


# Downsampling no finding images
random.seed(42)
nf_train = random.sample(nf_train, len(p_train))
nf_val = random.sample(nf_val, len(p_val))
nf_test=random.sample(nf_test,len(p_test))
print(len(nf_train),len(nf_val),len(nf_test))


# In[ ]:


train=e_train+p_train+nf_train
val=e_val+p_val+nf_val
test=e_test+p_test+nf_test
print(len(train),len(val),len(test))


# In[ ]:


# Check whether there are same patients in three parts
print(check_inter(train,val))
print(check_inter(train,test))
print(check_inter(val,test))


# In[ ]:


e_trainPath=get_path(images,e_train)
e_valPath=get_path(images,e_val)
e_testPath=get_path(images,e_test)

p_trainPath=get_path(images,p_train)
p_valPath=get_path(images,p_val)
p_testPath=get_path(images,p_test)

nf_trainPath=get_path(images,nf_train)
nf_valPath=get_path(images,nf_val)
nf_testPath=get_path(images,nf_test)


# In[ ]:


for path in tqdm(e_trainPath):
    dir_path='nofind_effusion_pneumothorax/train/1Effusion'
    create_dir(dir_path)
    shutil.copy(path, dir_path)

for path in tqdm(e_valPath):
    dir_path='nofind_effusion_pneumothorax/val/1Effusion'
    create_dir(dir_path)
    shutil.copy(path, dir_path)
    
for path in tqdm(e_testPath):
    dir_path='nofind_effusion_pneumothorax/test/1Effusion'
    create_dir(dir_path)
    shutil.copy(path, dir_path)
    
for path in tqdm(p_trainPath):
    dir_path='nofind_effusion_pneumothorax/train/2Pneumothorax'
    create_dir(dir_path)
    shutil.copy(path, dir_path)
    
for path in tqdm(p_valPath):
    dir_path='nofind_effusion_pneumothorax/val/2Pneumothorax'
    create_dir(dir_path)
    shutil.copy(path, dir_path)

for path in tqdm(p_testPath):
    dir_path='nofind_effusion_pneumothorax/test/2Pneumothorax'
    create_dir(dir_path)
    shutil.copy(path, dir_path)
    
for path in tqdm(nf_trainPath):
    dir_path='nofind_effusion_pneumothorax/train/0NoFinding'
    create_dir(dir_path)
    shutil.copy(path, dir_path)

for path in tqdm(nf_valPath):
    dir_path='nofind_effusion_pneumothorax/val/0NoFinding'
    create_dir(dir_path)
    shutil.copy(path, dir_path)
    
for path in tqdm(nf_testPath):
    dir_path='nofind_effusion_pneumothorax/test/0NoFinding'
    create_dir(dir_path)
    shutil.copy(path, dir_path)


# ### Prepare data for binary classification: No Finding  and Effusion. 
# Three subsets: training, validation and testing

# In[ ]:


e=class_names(df,'Effusion')
nf=class_names(df,'No Finding')
print(len(e),len(nf))


# In[ ]:


n_etrain = round(len(e)*0.6)
n_eval=round(len(e)*0.2)

e_train=e[0:n_etrain]
e_val=e[n_etrain:n_etrain+n_eval]
e_test=e[n_etrain+n_eval:n_etrain+n_eval*2]

print(len(e_train),len(e_val),len(e_test))


# In[ ]:


# There are overlap patient in e_val and e_test, extrange them
index1 = e_val.index('00022233_000.png')
index2 = e_test.index('00022245_002.png')

# Extrange them
e_val[index1], e_test[index2] = e_test[index2], e_val[index1]


# In[ ]:


# e_val.index('00022233_000.png')
# e_test.index('00022245_002.png')
# They have been extranged


# In[ ]:


extract_e=[item.split('_')[0] for item in e]
uni_e=set(extract_e)

extract_nf=[item.split('_')[0] for item in nf]
uni_nf=set(extract_nf)

difference = [item for item in uni_nf if item not in uni_e] #Filter out patient ids that are in nf but not in e
print(len(difference))


# In[ ]:


# Split remained no find images to three parts and make there are no same patients
random.seed(42)
sublist1 = random.sample(difference, len(e_train))
remaining_list = [item for item in difference if item not in sublist1]
sublist2 = random.sample(remaining_list, len(e_val))
remaining_list = [item for item in remaining_list if item not in sublist2]
sublist3 = random.sample(remaining_list, len(e_test))


# In[ ]:


nf_train,nf_val,nf_test=[],[],[]
for name in nf:
    if name.split('_')[0] in sublist1:
        nf_train.append(name)
    elif name.split('_')[0] in sublist2:
        nf_val.append(name)
    elif name.split('_')[0] in sublist3:
        nf_test.append(name)


# In[ ]:


# Downsampling no finding
random.seed(42)
nf_train = random.sample(nf_train, len(e_train))
nf_val = random.sample(nf_val, len(e_val))
nf_test=random.sample(nf_test,len(e_test))
print(len(nf_train),len(nf_val),len(nf_test))


# In[ ]:


e_trainPath=get_path(images,e_train)
e_valPath=get_path(images,e_val)
e_testPath=get_path(images,e_test)

nf_trainPath=get_path(images,nf_train)
nf_valPath=get_path(images,nf_val)
nf_testPath=get_path(images,nf_test)


# In[ ]:


for path in tqdm(e_trainPath):
    dir_path='nofind_effusion/train/1Effusion'
    create_dir(dir_path)
    shutil.copy(path, dir_path)

for path in tqdm(e_valPath):
    dir_path='nofind_effusion/val/1Effusion'
    create_dir(dir_path)
    shutil.copy(path, dir_path)
    
for path in tqdm(e_testPath):
    dir_path='nofind_effusion/test/1Effusion'
    create_dir(dir_path)
    shutil.copy(path, dir_path)
    
for path in tqdm(nf_trainPath):
    dir_path='nofind_effusion/train/0NoFinding'
    create_dir(dir_path)
    shutil.copy(path, dir_path)

for path in tqdm(nf_valPath):
    dir_path='nofind_effusion/val/0NoFinding'
    create_dir(dir_path)
    shutil.copy(path, dir_path)
    
for path in tqdm(nf_testPath):
    dir_path='nofind_effusion/test/0NoFinding'
    create_dir(dir_path)
    shutil.copy(path, dir_path)


# ### Prepare data for binary classification: No Finding and Pneumothorax. 
# Three subsets: training, validation and testing

# In[ ]:


p=class_names(df,'Pneumothorax')
nf=class_names(df,'No Finding')
print(len(p),len(nf))


# In[ ]:


n_train = round(len(p)*0.6)
n_val=round(len(p)*0.2)

p_train=p[0:n_train]
p_val=p[n_train:n_train+n_val]
p_test=p[n_train+n_val:n_train+n_val*2]

print(len(p_train),len(p_val),len(p_test))


# In[ ]:


# Exchange overlap patients' id
index1 = p_train.index('00017801_004.png')
index2 = p_val.index('00017942_002.png')

p_train[index1], p_val[index2] = p_val[index2], p_train[index1]


# In[ ]:


index1 = p_val.index('00024749_000.png')
index2 = p_test.index('00025082_007.png')

p_val[index1], p_test[index2] = p_test[index2], p_val[index1]


# In[ ]:


extract_p=[item.split('_')[0] for item in p]
uni_p=set(extract_p)

extract_nf=[item.split('_')[0] for item in nf]
uni_nf=set(extract_nf)

difference = [item for item in uni_nf if item not in uni_p] #Filter out patient ids that are in nf but not in e
print(len(difference))


# In[ ]:


# Split remained no find images to three parts and make there are no same patients
random.seed(42)
sublist1 = random.sample(difference, len(p_train))
remaining_list = [item for item in difference if item not in sublist1]
sublist2 = random.sample(remaining_list, len(p_val))
remaining_list = [item for item in remaining_list if item not in sublist2]
sublist3 = random.sample(remaining_list, len(p_test))


# In[ ]:


nf_train,nf_val,nf_test=[],[],[]
for name in nf:
    if name.split('_')[0] in sublist1:
        nf_train.append(name)
    elif name.split('_')[0] in sublist2:
        nf_val.append(name)
    elif name.split('_')[0] in sublist3:
        nf_test.append(name)


# In[ ]:


# Downsampling no finidng
random.seed(42)
nf_train = random.sample(nf_train, len(p_train))
nf_val = random.sample(nf_val, len(p_val))
nf_test=random.sample(nf_test,len(p_test))
print(len(nf_train),len(nf_val),len(nf_test))


# In[ ]:


p_trainPath=get_path(images,p_train)
p_valPath=get_path(images,p_val)
p_testPath=get_path(images,p_test)

nf_trainPath=get_path(images,nf_train)
nf_valPath=get_path(images,nf_val)
nf_testPath=get_path(images,nf_test)


# In[ ]:


for path in tqdm(p_trainPath):
    dir_path='nofind_pneumothorax/train/1Pneumothorax'
    create_dir(dir_path)
    shutil.copy(path, dir_path)
    
for path in tqdm(p_valPath):
    dir_path='nofind_pneumothorax/val/1Pneumothorax'
    create_dir(dir_path)
    shutil.copy(path, dir_path)

for path in tqdm(p_testPath):
    dir_path='nofind_pneumothorax/test/1Pneumothorax'
    create_dir(dir_path)
    shutil.copy(path, dir_path)

for path in tqdm(nf_trainPath):
    dir_path='nofind_pneumothorax/train/0NoFinding'
    create_dir(dir_path)
    shutil.copy(path, dir_path)

for path in tqdm(nf_valPath):
    dir_path='nofind_pneumothorax/val/0NoFinding'
    create_dir(dir_path)
    shutil.copy(path, dir_path)
    
for path in tqdm(nf_testPath):
    dir_path='nofind_pneumothorax/test/0NoFinding'
    create_dir(dir_path)
    shutil.copy(path, dir_path)


# ### Prepare data for binary classification: No Finding and Cardiomegaly. 
# Three subsets: training, validation and testing

# In[ ]:


c=class_names(df,'Cardiomegaly')
nf=class_names(df,'No Finding')
print(len(c),len(nf))


# In[ ]:


n_train = 657
n_val=218 

c_train=c[0:n_train]
c_val=c[n_train:n_train+n_val]
c_test=c[n_train+n_val:n_train+n_val*2]

print(len(c_train),len(c_val),len(c_test))


# In[ ]:


extract_c=[item.split('_')[0] for item in c]
uni_c=set(extract_c)

extract_nf=[item.split('_')[0] for item in nf]
uni_nf=set(extract_nf)

difference = [item for item in uni_nf if item not in uni_c] #Filter out patient ids that are in nf but not in e
print(len(difference))


# In[ ]:


# Split remained no find images to three parts and make there are no same patients
random.seed(42)
sublist1 = random.sample(difference, len(c_train))
remaining_list = [item for item in difference if item not in sublist1]
sublist2 = random.sample(remaining_list, len(c_val))
remaining_list = [item for item in remaining_list if item not in sublist2]
sublist3 = random.sample(remaining_list, len(c_test))


# In[ ]:


nf_train,nf_val,nf_test=[],[],[]
for name in nf:
    if name.split('_')[0] in sublist1:
        nf_train.append(name)
    elif name.split('_')[0] in sublist2:
        nf_val.append(name)
    elif name.split('_')[0] in sublist3:
        nf_test.append(name)


# In[ ]:


# Downsampling no finding
random.seed(42)
nf_train = random.sample(nf_train, len(c_train))
nf_val = random.sample(nf_val, len(c_val))
nf_test=random.sample(nf_test,len(c_test))
print(len(nf_train),len(nf_val),len(nf_test))


# In[ ]:


c_trainPath=get_path(images,c_train)
c_valPath=get_path(images,c_val)
c_testPath=get_path(images,c_test)

nf_trainPath=get_path(images,nf_train)
nf_valPath=get_path(images,nf_val)
nf_testPath=get_path(images,nf_test)


# In[ ]:


for path in tqdm(c_trainPath):
    dir_path='nofind_cardiomegaly/train/1Cardiomegaly'
    create_dir(dir_path)
    shutil.copy(path, dir_path)

for path in tqdm(c_valPath):
    dir_path='nofind_cardiomegaly/val/1Cardiomegaly'
    create_dir(dir_path)
    shutil.copy(path, dir_path)
    
for path in tqdm(c_testPath):
    dir_path='nofind_cardiomegaly/test/1Cardiomegaly'
    create_dir(dir_path)
    shutil.copy(path, dir_path)
    
for path in tqdm(nf_trainPath):
    dir_path='nofind_cardiomegaly/train/0NoFinding'
    create_dir(dir_path)
    shutil.copy(path, dir_path)

for path in tqdm(nf_valPath):
    dir_path='nofind_cardiomegaly/val/0NoFinding'
    create_dir(dir_path)
    shutil.copy(path, dir_path)
    
for path in tqdm(nf_testPath):
    dir_path='nofind_cardiomegaly/test/0NoFinding'
    create_dir(dir_path)
    shutil.copy(path, dir_path)

