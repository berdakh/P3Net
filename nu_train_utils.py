# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:11:14 2019 @author: berdakh.abibullaev
"""
import numpy as np
import time
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader
from nu_models import LSTM_Model, CNNLSTM
import pdb 
#%%   
def train_model(model, dset_loaders, dset_sizes, criterion, optimizer, dev,
                lr_scheduler=None, num_epochs=50, verbose=2):    
      """
      Method to train a PyTorch neural network with the given parameters for a
      certain number of epochs. Keeps track of the model yielding the best validation
      accuracy during training and returns that model before potential overfitting
      starts happening. Records and returns training and validation losses and accuracies over all
      epochs.

      Args:
          model (torch.nn.Module): The neural network model that should be trained.

          dset_loaders (dict[string, DataLoader]): Dictionary containing the training
              loader and test loader: {'train': trainloader, 'val': testloader}
          dset_sizes (dict[string, int]): Dictionary containing the size of the training
              and testing sets. {'train': train_set_size, 'val': test_set_size}

          criterion (PyTorch criterion): PyTorch criterion (e.g. CrossEntropyLoss)
          optimizer (PyTorch optimizer): PyTorch optimizer (e.g. Adam)

          lr_scheduler (PyTorch learning rate scheduler, optional): PyTorch learning rate scheduler
          num_epochs (int): Number of epochs to train for
          verbose (int): Verbosity level. 0 for none, 1 for small and 2 for heavy printouts
      """     

      start_time = time.time()
      best_model, best_acc = model, 0.0 

      train_losses, val_losses, train_accs, val_accs  = [], [], [], []
      train_labels, val_labels = [], []

      for epoch in range(num_epochs):     
          if verbose > 1: print('Epoch {}/{}'.format(epoch+1, num_epochs))          
          ypred_labels, ytrue_labels = [], [] 

          # there are two phases [Train and Validation]           
          for phase in ['train', 'val']:
              if phase == 'train':
                  if lr_scheduler: optimizer = lr_scheduler(optimizer, epoch)
                  model.train(True)  # Set model to training mode
              else:
                  model.train(False)  # Set model to evaluate mode                   
              running_loss, running_corrects = 0.0, 0.0 
              # Iterate over mini-batches
              batch = 0
              for data in dset_loaders[phase]:
                  input, label = data
                  inputs = input.to(dev)
                  labels = label.to(dev) 
                  model  = model.cuda()
                  optimizer.zero_grad()                                      
                  
                  if isinstance(model, LSTM_Model) or isinstance(model, CNNLSTM):
                      hidden = model.init_hidden(len(inputs)) # Zero the hidden state   
                      preds = model(inputs, hidden) # Forward pass                      
                  else:  
                      # pdb.set_trace()
                      preds = model(inputs) # Forward pass  
                  # Calculate the loss 
                  loss = criterion(preds, labels)                  
                  # Backpropagate & weight update 
                  if phase == 'train':
                      loss.backward()
                      optimizer.step()                       
                  # store batch performance 
                  running_loss += loss.item()
                  preds_classes = preds.data.max(1)[1]
                  running_corrects += torch.sum(preds_classes == labels.data) 
                  
                  ytrue_labels.append(labels.data.cpu().detach().numpy())
                  ypred_labels.append(preds_classes.cpu().detach().numpy())            
                  
                  batch += 1
                  
              epoch_loss = running_loss / dset_sizes[phase]
              epoch_acc = running_corrects.cpu().numpy()/dset_sizes[phase]         

              if phase == 'train':
                  train_losses.append(epoch_loss)
                  train_accs.append(epoch_acc)                   
                  train_labels.append(dict(ypred = ypred_labels, ytrue = ytrue_labels))
                  
              else: # val
                  val_losses.append(epoch_loss)
                  val_accs.append(epoch_acc)                
                  val_labels.append(dict(ypred = ypred_labels, ytrue = ytrue_labels))

              if verbose > 1: print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
              # Deep copy the best model using early stopping
              if phase == 'val' and epoch_acc > best_acc:
                  best_acc = epoch_acc
                  best_epoch = epoch 
                  best_model = copy.deepcopy(model)           

      time_elapsed = time.time() - start_time  
      
      # ytrue and ypred from the best model during the training            
      def best_epoch_labels(train_labels, best_epoch):        
        for jj in range(len(train_labels[best_epoch]['ypred'])-1):    
            if jj == 0: 
              ypred = train_labels[best_epoch]['ypred'][jj]              
              ytrue = train_labels[best_epoch]['ytrue'][jj]                  
            ypred = np.concatenate([ypred, train_labels[best_epoch]['ypred'][jj+1]])
            ytrue = np.concatenate([ytrue, train_labels[best_epoch]['ytrue'][jj+1]])            
        return ypred, ytrue
      
      ytrain_best = best_epoch_labels(train_labels, best_epoch)
      yval_best = best_epoch_labels(val_labels, best_epoch)      
      
      info = dict(ytrain = ytrain_best, yval = yval_best, 
                  best_epoch = best_epoch, best_acc = best_acc)
      
      if verbose > 0:
          print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
          print('Best val Acc: {:4f}'.format(best_acc)) 
          print('Best Epoch :', best_epoch+1)          
      return best_model, train_losses, val_losses, train_accs, val_accs, info

