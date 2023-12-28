'''
This script is supporting analysis on subset of imagenet called ImageNet20 obtained
from Kaggle: https://www.kaggle.com/datasets/shahnazari/imagenet20

'''


import math
import time
import numpy as np
import argparse
import yaml
import copy

# Pytorch package
import torch
import torch.nn as nn
import torchvision
from torchvision import models, datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


from datetime import datetime
from data_utilities import *
from model_utilities import *
from fooling_utilities import *

print (torchvision.__version__)
#reading in the arguments and inputs

parser = argparse.ArgumentParser(description='RobustNet')
parser.add_argument('--config', default='./configs/config_robustnet.yaml')

global args
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader = yaml.FullLoader)

for key in config:
    for k, v in config[key].items():
        setattr(args, k, v)




#parameters and hyperparameters

b_size = args.batch_size  #training batch size
val_b_size = args.validation_batch_size #validation batch_size
test_b_size = args.test_batch_size #test_batch_size

if args.debug == True:
    num_classes = 3
else:
    num_classes = 20 #20 classes in subset imagenet

#assigning device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: %s" % device)



#importing subset imagenet20 data

if args.debug == True:
    path = './datasets/debug_ImageNet'
else: 
    path = './datasets/subImageNet'

if args.is_fooling:
    print ('Using fooling images for training.')
    fool_model = models.resnet18(pretrained = True)
    num_ftrs = fool_model.fc.in_features
    fool_model.fc = nn.Linear(num_ftrs, num_classes)
    if args.debug ==  True:
        weights = torch.load('./logs/ResNet18_FCUpdate_Pretrained_Dummy_20_0.005_0.95/resnet18FCUpdate_Pretrained_Dummy_20_0.005_0.95.pth')
    else:
        weights = torch.load('./logs/ResNet18_FCUpdate_Pretrained_20_0.005_0.95/resnet18FCUpdate_Pretrained_20_0.005_0.95.pth')
    
    fool_model.load_state_dict(weights)
    fool_model = fool_model.to(device)
    
    sub20_imagenet = load_subset20_imagenet_fooling(fool_model,path,args.fooling_lr,args.fool_iterations,dtype=np.float32, subtract_mean=False)
else:
    sub20_imagenet = load_subset20_imagenet(path, dtype=np.float32, subtract_mean=False)

print (sub20_imagenet['X_train'].shape)
print (sub20_imagenet['X_train'][0,:,:,:])


#creating dataloader for training, validation, and test sets

transform = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   
]) #if image augmentation is needed, add custom transformations

'''transform = transforms.Compose([
   transforms.AugMix(severity=3,mixture_width=3,all_ops=False),
   transforms.ToTensor(),
   transforms.Resize((224,224)),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   
]) #if image augmentation is needed, add custom transformations
'''

'''transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])'''

target_transform = None #edit if the labels need additional processing

#loading data for training  and validation on  subset  imagenet20

training_dataset = CustomDataset(sub20_imagenet['X_train'],sub20_imagenet['y_train'], transform, target_transform)
validation_dataset = CustomDataset(sub20_imagenet['X_val'],sub20_imagenet['y_val'], transform, target_transform)

#loading data loader

train_loader = DataLoader(training_dataset, batch_size= b_size, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size= val_b_size, shuffle=True)




dataloaders={'train':train_loader,'val':val_loader}


#loading and initializaing model (feature extracting)

if args.type == 'ResNet18':
    file_name = str(args.type)+'_pre_trained_' +str(args.pre_trained)+'_feature_extract_'+str(args.feature_extract) + args.details
    model_feature_extract = models.resnet18(pretrained = args.pre_trained)
else:
    file_name = str(args.type)+'_pre_trained_' +str(args.pre_trained)+'_feature_extract_'+str(args.feature_extract)
    #use custom model
    
    
model_feature_extract = model_feature_extract.to(device)

feature_extract = args.feature_extract # the model freezes the kernel weights and only updates the fc layer

set_parameter_requires_grad(model_feature_extract, feature_extract)

num_ftrs = model_feature_extract.fc.in_features
model_feature_extract.fc = nn.Linear(num_ftrs, num_classes) 


print (model_feature_extract)
print (model_feature_extract.layer4[0].conv1.weight.size())
print (model_feature_extract.layer4[0].conv1)



#pre-processing of images



#applying model

criterion = nn.CrossEntropyLoss()
params_to_update = model_feature_extract.parameters()
optimizer = torch.optim.SGD(params_to_update, lr=args.learning_rate, momentum=args.momentum)

train_model_output = train_model(model_feature_extract, dataloaders, criterion, optimizer, num_epochs=args.epochs)

#plotting the results across epochs
#plot_results(train_model_output[5].cpu(),train_model_output[3].cpu(),'Loss',args.details)
#plot_results(train_model_output[4].cpu(),train_model_output[2].cpu(),'Accuracy',args.details)

#reviewing results

#########  importing and analyzing imagenet subset20 corrupt data  ##################
print (f"Validation accuracy: {train_model_output[1]}")
error_on_clean = 1.0 - train_model_output[1].item()
rn18_error_on_clean = 0.085
rn18_error_on_ctype = {}

if args.debug == True:
    test_path = './datasets/debug_ImageNetC'
else:
    test_path = './datasets/ImageNetC'

if args.debug == True:
    corruption_type = ['brightness','frost']
else:
    corruption_type = ['brightness','contrast','defocus_blur','elastic_transform','fog','frost','gaussian_noise','glass_blur','impulse_noise','jpeg_compression','motion_blur','pixelate','shot_noise','snow','zoom_blur','speckle_noise',
    'gaussian_blur','spatter','saturate']
    
severity_level = ['1','2','3','4','5']

track_data={'ctype':[],'sev_level':[],'Acc':[],'Error_rate':[]}
mce_data={'ctype':[],'ctype_error':[],'rel_corr_error':[]}

mce_error = 0.0
mean_rce_error = 0.0

rn18_error=[0.1338,0.4722,0.408,0.2488,0.3914,0.431,0.4874,0.4118,0.5458,0.2546,0.415,0.3134,0.5066,0.4738,0.4462,0.4126,0.375,0.311,0.2094]

for i,ctype in enumerate(corruption_type):
    rn18_error_on_ctype[ctype]= rn18_error[i]


for ctype in corruption_type:
    
    sum_error = 0.0
    rce_error = 0.0
    
    for slevel in severity_level:
        sub20_corrupt_imagenet = load_sub20_imagenet_corrupt(test_path, sub20_imagenet['id_labels'], sub20_imagenet['mean_image'], severity = slevel, typeof = ctype,subtract_mean=False)
        test_corrupt_dataset = CustomDataset(sub20_corrupt_imagenet['X'],sub20_corrupt_imagenet['y'], transform, target_transform)
        corrupt_test_loader = DataLoader(test_corrupt_dataset, batch_size= test_b_size, shuffle=True) #imagenet corrupted data
        test_acc,_ = evaluate_model(train_model_output[0],corrupt_test_loader)
        
        print (f"Test accuracy for corruption type: {ctype} and severity level: {slevel} : {test_acc}")
        track_data['ctype'].append(ctype)
        track_data['sev_level'].append(slevel)
        track_data['Acc'].append(test_acc.item())
        track_data['Error_rate'].append(1.0-test_acc.item())
        
        
        sum_error+= (1.0-test_acc.item())/5
        
    rce_error = (sum_error - error_on_clean)/(rn18_error_on_ctype[ctype]-rn18_error_on_clean)
    mean_rce_error+= rce_error/len(corruption_type)
    
    sum_error = sum_error/(rn18_error_on_ctype[ctype])
    
    mce_data['ctype'].append(ctype)
    mce_data['ctype_error'].append(sum_error)
    mce_data['rel_corr_error'].append(rce_error)
    
    mce_error += sum_error/len(corruption_type)

print (f"MCE error: {mce_error}")



mce_data['ctype'].extend(['Clean','MCE'])
mce_data['ctype_error'].extend([error_on_clean,mce_error])
mce_data['rel_corr_error'].extend([0,mean_rce_error])
        
        
track_df = pd.DataFrame.from_dict(track_data)
mce_df = pd.DataFrame.from_dict(mce_data)

now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")


track_df.to_csv('./logs/' + file_name + '_' + now +'_rawdata.csv',index = False)
mce_df.to_csv('./logs/' + file_name + '_' + now + '_mce.csv', index = False)          

#saving best model
torch.save(train_model_output[0].state_dict(), './logs/' + args.type.lower() + args.details + '.pth')