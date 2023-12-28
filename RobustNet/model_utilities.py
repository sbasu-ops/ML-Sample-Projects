import time
import torch
import torch.nn as nn
import torchvision
from torchvision import models, datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    '''
    The function has been adpated with minor changes from the turorial provided
    at https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks
    
    '''
    
    
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    
    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    count = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            iterate = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                model = model.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                if iterate%20==0:
                    if phase == 'train':
                        print (f"Running batch: {iterate} out of {len(dataloaders[phase].dataset)/labels.shape[0]} in epoch: {epoch} for {phase}")
                    else:
                        print (f"Running batch: {iterate} out of {len(dataloaders[phase].dataset)/labels.shape[0]} in epoch: {epoch} for {phase}")
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                  
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                iterate+=1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            
            if phase == 'val' and epoch_acc <= best_acc:
                count+=1
            
            if phase == 'val' and epoch_acc > best_acc:
                count = 0
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase ==  'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
        
        if count >=10:
            print (f"Early exit at epoch: {epoch}")
            break
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, val_acc_history, val_loss_history, train_acc_history, train_loss_history
    
    
def set_parameter_requires_grad(model, feature_extracting):
    '''
    
    The function has been adpated with minor changes from the turorial provided
    at https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks
   
    '''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def evaluate_model(model,dataloaders):
    model.eval()
    running_corrects = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)
        model = model.to(device)
        
        outputs = model(inputs)        
        _, preds = torch.max(outputs, 1)
        
        running_corrects += torch.sum(preds == labels.data)
       
    
    eval_acc = running_corrects.double() / len(dataloaders.dataset)
    
    return eval_acc,preds
    
    
    
