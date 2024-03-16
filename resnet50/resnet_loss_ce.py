#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import glob
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights
import torchvision
import pathlib
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn.functional as F
import copy
import sys
import seaborn as sns
from scipy.special import softmax
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.backends.backend_pdf import PdfPages
import time
import argparse


# In[ ]:


parser = argparse.ArgumentParser(description='Script for training and evaluating Resnet50 model with cross entropy loss')
parser.add_argument('--path',type=str, required=True, help="Path of data")
parser.add_argument('--gpu',type=int, default= 0, help = 'Specify a gpu if there are more than one')
parser.add_argument('--nclass',type=int, required=True, help = 'Specify number of classes for the classification task, must be 2 or 3')

args = parser.parse_args()


# In[ ]:


start_time = time.time() # Record the start time


# In[3]:


gpu_index = args.gpu
torch.cuda.set_device(gpu_index)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[4]:


transformer=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),  
    transforms.Normalize([0.5,0.5,0.5], 
                        [0.5,0.5,0.5])
])


# ### Define data path

# In[2]:


train_path = args.path+'train'
valid_path = args.path+'val'
test_path= args.path+'test'

train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformer),
    batch_size=32, shuffle=True
)
valid_loader=DataLoader(
    torchvision.datasets.ImageFolder(valid_path,transform=transformer),
    batch_size=32, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transformer),
    batch_size=32, shuffle=True
)


# In[ ]:


root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

print(classes)


# In[7]:


# Calculating the size of training and testing images
train_count=len(glob.glob(train_path+'/**/*.png'))
valid_count=len(glob.glob(valid_path+'/**/*.png'))
test_count=len(glob.glob(test_path+'/**/*.png'))

print(train_count,valid_count,test_count)


# In[9]:


model=models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_classes=args.nclass
model.fc = torch.nn.Linear(2048, num_classes)
# for param in model.parameters():
#   param.requires_grad = True
model.to(device)


# In[10]:


#Optmizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
loss_function=nn.CrossEntropyLoss()
lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, min_lr=1e-10,verbose=True)


# In[23]:


num_epochs=100

# Model training and saving best model
best_accuracy=0.0
no_improvement_epochs = 0
model_path='best_model.pt'

for epoch in range(num_epochs):
    
    #Evaluation and training on training dataset
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    
    for i, (images,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            
        optimizer.zero_grad()
        
        outputs=model(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        
        train_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))
        
    train_accuracy=train_accuracy/train_count
    train_loss=train_loss/train_count
    
    
    # Evaluation on validation dataset
    model.eval()
    test_loss=0.0
    test_accuracy=0.0
    for i, (images,labels) in enumerate(valid_loader):
        if torch.cuda.is_available():
          images=Variable(images.cuda())
          labels=Variable(labels.cuda())
            
        outputs=model(images)

        loss=loss_function(outputs,labels)
        test_loss+= loss.cpu().data*images.size(0)

        _,prediction=torch.max(outputs.data,1)
        test_accuracy+=int(torch.sum(prediction==labels.data))
    
    test_accuracy=test_accuracy/valid_count
    test_loss=test_loss/valid_count
    
    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test loss: '+str(test_loss)+' Test Accuracy: '+str(test_accuracy))
    
    #Save the best model
    if test_accuracy>best_accuracy:
        torch.save(model, model_path)
        best_accuracy=test_accuracy
    else:
        no_improvement_epochs+=1
    
    if no_improvement_epochs>=30:
        print('No improvement for specific epochs. Stopping the training.')
        break
        
    lr_scheduler.step(test_loss)


# In[ ]:


# Record the end time
end_time = time.time() 
execution_time_seconds = end_time - start_time
execution_time_minutes = execution_time_seconds / 60


# In[33]:


model = torch.load(model_path)
model.eval()

y_true,y_pred, y_pred_probabs=[],[],[]

for i, (images,labels) in enumerate(valid_loader):
  if torch.cuda.is_available():
    images=Variable(images.cuda())
    labels=Variable(labels.cuda())

  outputs=model(images)
  _,prediction=torch.max(outputs.data,1)

  y_true.extend(labels.tolist())
  y_pred.extend(prediction.tolist())
  y_pred_probabs.extend(outputs.tolist())

print(classification_report(y_true,y_pred, digits=3))


# In[ ]:


# Evaluate on unseen test data
y_true2,y_pred2, y_pred_probabs2=[],[],[]

for i, (images,labels) in enumerate(test_loader):
  if torch.cuda.is_available():
    images=Variable(images.cuda())
    labels=Variable(labels.cuda())

  outputs=model(images)
  _,prediction=torch.max(outputs.data,1)

  y_true2.extend(labels.tolist())
  y_pred2.extend(prediction.tolist())
  y_pred_probabs2.extend(outputs.tolist())

print(classification_report(y_true2,y_pred2, digits=3))


# **AUC**

# In[ ]:


def calculate_auc(y_true, y_pred_probabs):
    if args.nclass == 2:
        softmax_preds = softmax(y_pred_probabs, axis=1)
        positive_probabilities = softmax_preds[:, 1]
        return roc_auc_score(y_true, positive_probabilities)

    elif args.nclass == 3:
        y_pred_probabs_softmax = torch.softmax(torch.tensor(y_pred_probabs), dim=1)
        y_pred_probabs_softmax = y_pred_probabs_softmax.numpy()
        return roc_auc_score(y_true, y_pred_probabs_softmax, multi_class='ovr')

y_pred_probabs = np.array(y_pred_probabs)
y_pred_probabs2 = np.array(y_pred_probabs2)

# Calculate AUC for validation set
auc_val = calculate_auc(y_true, y_pred_probabs)

# Calculate AUC for testing set
auc_test = calculate_auc(y_true2, y_pred_probabs2)


# **Save results**

# In[ ]:


original_stdout = sys.stdout
with open("results.txt", "w") as f:
    sys.stdout = f
    print("Execution Time:", execution_time_minutes, "minutes")
    print(classification_report(y_true,y_pred, digits=3))
    print("AUC Score:", auc_val)
    print('-'*30)
    print("Metrics of unseen test data: ")
    print(classification_report(y_true2,y_pred2, digits=3))
    print("AUC Score:", auc_test)
sys.stdout = original_stdout

