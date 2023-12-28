from __future__ import print_function
from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from matplotlib.pyplot import imread
from skimage.transform import rescale, resize
import platform
import torch
import pandas as pd
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import matplotlib

from PIL import Image

from torchvision.transforms import ToTensor
from torch.autograd import Variable

import torch
import torch.nn as nn
import torchvision
from torchvision import models, datasets, transforms
import torch.optim as optim


'''def plot_images(image,title,string):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.title(title+'_'+string)
    plt.savefig('./datasets/ImageNetFool20/train/' + title + '/'+ string + '.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    #plt.show()
    return'''
    
def plot_images(image,title,string):
    file_name = './datasets/ImageNetFool5/train/' + title + '/'+ string + '.jpeg'
    print (file_name)
    
    image = (image-np.min(image))/(np.max(image)-np.min(image))
 
    print (image.shape)
   
    result = Image.fromarray((image*255.0).astype(np.uint8),'RGB')
    result.save(file_name)
    
    return

def fooling_image(model,X_initial,y_initial,y_target,lrate,max_iter):
    '''
    This code is based on the helper code for fooling image provided in the assignment
    which required the students to complete tne main algorithm.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    X_train_init = np.zeros((1, 3, X_initial.shape[1], X_initial.shape[2]), dtype=np.float32)
    
    X_train_init[0] = X_initial
  
    
    X_train_init = torch.from_numpy(X_train_init).to(device)
    #print (X_train_init.shape)
    # Initialize our fooling image to the input image, and wrap it in a Variable.
    X_fooling = X_train_init.clone()
    X_fooling_var = Variable(X_fooling, requires_grad=True)
    #print (y_initial)
    #print (y_target)
    for i in range(max_iter):
        if (i==98):
            print ('Max iterations reached.')
        X_fooling_var.retain_grad()
        out = model(X_fooling_var)
        S = out[0,y_target]
        S.backward()
            
        gradient = ((X_fooling_var.grad)/torch.norm(X_fooling_var.grad,p=2))
           
        X_fooling_var = X_fooling_var + lrate*gradient
            
        #check prediction
        #print (out.data.max(1)[1][0]) 
        if y_target == out.data.max(1)[1][0]:
            break
    
    #print (i)
    
        
    X_fooling = X_fooling_var.data
    
    X_fooling = torch.squeeze(X_fooling)
    
    X_fooling  = X_fooling.cpu().detach().numpy()
    
    
    return X_fooling.transpose(1,2,0)




#This code is modifies/updates the helper code provided by the instruction team 
def load_subset20_imagenet_fooling(fool_model,path,lrate,max_iter,dtype=np.float32, subtract_mean=False):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.

    Inputs:
    - fool_model: Pretrained model used for the fooling image creation
    - path: String giving path to the directory to load.
    - lrate: learning rate for fooling image gradient ascent
    - max_iter: Maximum number of iterations
    - dtype: numpy datatype used to load the data.
    - subtract_mean: Whether to subtract the mean training image.

    Returns: A dictionary with the following entries:
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 3, 64, 64) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 3, 64, 64) array of validation images
    - y_val: (N_val,) array of validation labels
    - mean_image: (3, 64, 64) array giving mean training image
    """
    #identifying classes in subset ImageNet20
    subset_wid = []
    wid_list = os.listdir(os.path.join(path,'train'))
    for _,wid in enumerate(wid_list):
        subset_wid.append(str(wid))
    print (subset_wid)
    
    # First load wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    #wnid_to_label = {wnid: i for i, wnid in enumerate(wnids) if wnid in subset_wid}
    wnid_to_label = {wnid: i for i, wnid in enumerate(subset_wid)}
    print (f"final labels: {wnid_to_label}")
    
   
    # Use words.txt to get names for each class
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in subset_wid]
    print (f"class names: {class_names}")
    
    # Next load training data.
    X_train = []
    y_train = []
    
    
    for i, wnid in enumerate(subset_wid):
        if (i + 1) % 5 == 0:
            print('loading training data for synset %d / %d' % (i + 1, len(subset_wid)))
        
        # To figure out the filenames we need to open the boxes file
        images_file = os.listdir(os.path.join(path, 'train', wnid))
        
        num_images = len(images_file)
        y_train.extend([wnid_to_label[wnid]]*num_images)
        
        #X_train_block = np.zeros((num_images, 3, 224, 224), dtype=dtype)
        
        
        for j, im_file in enumerate(images_file):
            #print (im_file)
            y_target = wnid_to_label[wnid]
            img_file = os.path.join(path, 'train', wnid, im_file)
            
            img = imread(img_file).astype(dtype)/255.0
            
            
            if img.ndim == 2:
                # grayscale file
                #print (im_file)
                #img = resize(img,(224,224))
                img.shape = (img.shape[0], img.shape[1], 1)
            #else:
                #img = resize(img,(224,224))
            
            
                
                
            
            while y_target == wnid_to_label[wnid]:
                y_target = random.randint(0,len(subset_wid)-1)
            
            X_fooling = fooling_image(fool_model,img.transpose(2, 0, 1),wnid_to_label[wnid],y_target,lrate,max_iter)
            
            
            plot_images(X_fooling,wnid,class_names[y_target][0]+'_'+str(j))
            
           
            
        #X_train.append(X_train_block)
  

    # Convert to numpy array from list
    #X_train = np.concatenate(X_train,axis=0)
    y_train = np.array(y_train,dtype=np.int64)
    
    #print (X_train.shape)
    #print (y_train.shape)
    #print (y_train[1290:1310])
    
    # Next load validation data
            
    X_val = []
    y_val = []
    for i, wnid in enumerate(subset_wid):
        if (i + 1) % 5 == 0:
            print('loading validation data for synset %d / %d' % (i + 1, len(subset_wid)))
        
        # To figure out the filenames we need to open the boxes file
        val_images_file = os.listdir(os.path.join(path, 'val', wnid))
        
        val_num_images = len(val_images_file)
        y_val.extend([wnid_to_label[wnid]]*val_num_images)
        X_val_block = np.zeros((val_num_images, 3, 224, 224), dtype=dtype)
   
        for j, im_file in enumerate(val_images_file):
            print ('In Validation')
            #print (im_file)
            img_file = os.path.join(path, 'val', wnid, im_file)
            img = imread(img_file).astype(dtype)/255.0
            img = resize(img,(224,224))
            if img.ndim == 2:
                # grayscale file
                img.shape = (img.shape[0], img.shape[1], 1)
           
            X_val_block[j] = img.transpose(2, 0, 1)
            
        X_val.append(X_val_block)
  

    # Convert to numpy array from list
    X_val = np.concatenate(X_val,axis=0)
    y_val = np.array(y_val,dtype=np.int64)
    
    print (X_val.shape)
    print (y_val.shape)
    print (y_val[40:110])
   
    mean_image = X_train.mean(axis=0)
    if subtract_mean:
        X_train -= mean_image[None]
        X_val -= mean_image[None]

    return {
        'id_labels': wnid_to_label,
        'class_names': class_names,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'mean_image': mean_image,
    }