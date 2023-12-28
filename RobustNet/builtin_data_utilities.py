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
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import matplotlib


'''def create_dataset(path,transforms,b_size,shuffle):
    selected_dataset = datasets.ImageFolder(root = path,transform = transforms)
    dataset_loader = DataLoader(selected_dataset, batch_size = b_size,shuffle=shuffle)
    return dataset_loader'''
    

def create_dataset(transforms,b_size,sfle,**kwargs):
    concat_dataset = []
    
    for key,value in kwargs.items():
        selected_dataset = datasets.ImageFolder(root = value,transform = transforms)
        concat_dataset.append(selected_dataset)
    
    combined_dataset = torch.utils.data.ConcatDataset(concat_dataset)
    dataset_loader = DataLoader(combined_dataset, batch_size = b_size,shuffle=sfle)
    return dataset_loader
    



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
    
    

def plot_images(image,title,string):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.title(title+'_'+string)
    plt.savefig('./datasets/ImageNetFool20/train/' + title + '/'+ string + '.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    #plt.show()
    return