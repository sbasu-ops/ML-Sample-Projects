'''
This script is supporting analysis on subset of imagenet called ImageNet20 obtained
from Kaggle: https://www.kaggle.com/datasets/shahnazari/imagenet20

'''



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

from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib



#This code is modifies/updates the helper code provided by the instruction team 
def load_subset20_imagenet(path, dtype=np.float32, subtract_mean=False):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.

    Inputs:
    - path: String giving path to the directory to load.
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
        
        X_train_block = np.zeros((num_images, 3, 224, 224), dtype=dtype)
   
        for j, im_file in enumerate(images_file):
            #print (im_file)
            img_file = os.path.join(path, 'train', wnid, im_file)
            
            img = imread(img_file).astype(dtype)/255.0
            
            
            if img.ndim == 2:
                # grayscale file
                #print (im_file)
                img = resize(img,(224,224))
                img.shape = (img.shape[0], img.shape[1], 1)
            else:
                img = resize(img,(224,224))
           
            X_train_block[j] = img.transpose(2, 0, 1)
            
        X_train.append(X_train_block)
  

    # Convert to numpy array from list
    X_train = np.concatenate(X_train,axis=0)
    y_train = np.array(y_train,dtype=np.int64)
    
    print (X_train.shape)
    print (y_train.shape)
    print (y_train[1290:1310])
    
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



def image_info_extraction(path,corruption_type,class_labels,severity_level,info_tracking,X_old,y_old):
    '''
    Extract image information in numpy tensors for selected corruption type and severity level
    Inputs:
    - path: String giving path to the directory to load.
    - corruption_type: type of corruption. 
    - class_labels: original class labels from the tiny imagenet dataset.
    - severity_level: severity of corruption. Select the folder with that severity.
    - info_tracking: Dictionary tracking corruption type, severity level, and word id for processed image
    - X_old: The image tensor holding already downloaded image information.
    - y_old: The label tensor holding labels of already processed images
   
   

    Returns: A dictionary with the following entries:
    - X_old: Updated tensor holding images of given corruption type and severity level
    - y_old: Updated array of labels
    - info_tracking: Updated dictionary tracking corruption type, severity level, and word id for processed image
    
    
    '''
    
    wid_list = os.listdir(os.path.join(path,corruption_type,severity_level))
    for _,wid in enumerate(wid_list):
        
        target = class_labels[wid]
        img_list = os.listdir(os.path.join(path,corruption_type,severity_level,wid))
        
        y_old.extend([target]*len(img_list))
        info_tracking['corruption'].extend([corruption_type]*len(img_list))
        info_tracking['severity'].extend([severity_level]*len(img_list))
        info_tracking['wid'].extend([wid]*len(img_list))
        info_tracking['labels'].extend([class_labels[wid]]*len(img_list))
        
        for _,img_name in enumerate(img_list):
        
            img_file = os.path.join(path,corruption_type,severity_level,wid,img_name)
            img = imread(img_file).astype(np.float32)/255.0

            if img.ndim == 2:
                # grayscale file
                img.shape = (img.shape[0], img.shape[0], 1)
            X_old.append(img.transpose(2,0,1))
    return X_old,y_old,info_tracking  
    

def load_sub20_imagenet_corrupt(path, class_labels, mean_image, severity = '3', typeof = 'all',subtract_mean=False):
    
    """
    Load TinyImageNet-C. .

    Inputs:
    - path: String giving path to the directory to load.
    - class_labels: original class labels from the training set of imagenet20 dataset.
    - mean_image: mean image from the training set of imagenet20 dataset.
    - severity: severity of corruption. Select the folder with that severity. If all, select all levels of severity.
    - typeof: type of corruption. If all, select all types of corruption
    - subtract_mean: Whether to subtract the mean training image.

    Returns: A dictionary with the following entries:
    - X: (N_tr, 3, 64, 64) array of images
    - y: (N_tr,) array of label
    - class_info: details about corruption, severity level, and labels
    """
    
  
    
    corruption_type = ['brightness','contrast','defocus_blur','elastic_transform','fog','frost','gaussian_noise','glass_blur','impulse_noise','jpeg_compression','motion_blur','pixelate','shot_noise','snow','zoom_blur','speckle_noise',
    'gaussian_blur','spatter','saturate']
    severity_level =['1','2','3','4','5']
    info_details={'corruption':[],'severity':[],'wid':[], 'labels':[]}
    
    X =[]
    y= []
    
    if typeof == 'all':
        for ctype in corruption_type:
            if severity =='all':
                for slevel in severity_level:
                     X,y,info_details = image_info_extraction(path,ctype,class_labels,slevel,info_details,X,y)
                            
            else:
                
                X,y,info_details = image_info_extraction(path,ctype,class_labels,severity,info_details,X,y)
            
    else:
        if severity=='all':
            for slevel in severity_level:
                X,y,info_details = image_info_extraction(path,typeof,class_labels,slevel,info_details,X,y)
            
        else:
            X,y,info_details = image_info_extraction(path,typeof,class_labels,severity,info_details,X,y)
    
    X  =  np.array(X,dtype=np.float32)
    y  = np.array(y,dtype=np.int64)
    
    if subtract_mean:
        X -= mean_image[None]
    
    
    return {
     'X': X,
     'y': y,
     'class_info':info_details,
    }
    

def plot_results(train_list,val_list,str_type,mod_type):

    f, ax = plt.subplots()
    f.set_size_inches(10,5)

    plt.plot(np.arange(len(train_list)),train_list,'ko', label='Training '+str_type)
    plt.plot(np.arange(len(val_list)),val_list,'rd',label='Validation '+str_type)

    plt.xlabel("Epochs",fontsize='large')
    plt.ylabel(str_type,fontsize='large')

    #plt.xlim(50,80)
    #plt.ylim(58,62)

    plt.rcParams.update({'font.size':12})
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)

    plt.title(mod_type + ' ' + str_type)

    ax.legend(loc='best',fontsize='large')
    plt.savefig('./logs/'+ mod_type + '_' + str_type +  '.png', dpi=300, bbox_inches='tight')

    plt.show()
    return








class CustomDataset(Dataset):
   
   def __init__(self, X, y, transform = None, target_transform = None):
        self.image = torch.from_numpy(X)
        self.labels = torch.from_numpy(y).to(dtype=torch.long)
        
        self.transform = transform
        self.target_transform = target_transform
        
   def __len__(self):
        return len(self.labels)
        
   def __getitem__(self,idx):
        image = self.image[idx,:,:,:]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return  image,label
