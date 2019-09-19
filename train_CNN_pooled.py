# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:03:25 2019
@author: Berdakh

This script can be used to train CNN model on pooled data.
"""

import torch 
import itertools
import pandas as pd 
import pickle 

from nu_data_loader import getTorch, EEGDataLoader
from nu_train_utils import train_model   
from nu_models import CNN2D

# to get a torch tensor 
get_data = getTorch.get_data 

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Your GPU device name :', torch.cuda.get_device_name())    

#%%  
'''
dname is a dictionary containing dataset names to be loaded from
the current directory

The following files represent the ERP datasets referred in the paper as:

    NU data   = 'data_allsubjects.pickle',
    EPFL data = 'EPFLP300.pickle'
    BNCI data ='TenHealthyData.pickle'
    ALS data  ='ALSdata.pickle'
'''

dname = dict(nu = 'data_allsubjects.pickle', 
             epfl = 'EPFLP300.pickle',  
             ten = 'TenHealthyData.pickle',
             als = 'ALSdata.pickle')

#%% Hyperparameter settings
num_epochs = 100 
learning_rate = 1e-3
weight_decay = 1e-4  
batch_size = 64 
verbose = 2

#%%
# one should run this script twice with ConvDown = True or False to have different convolutional layer patterns
# as defined below by params dictionary. 

ConvDOWN = True   # change this option 

#%% The main loop starts here 
# for each dataset in dname train CNN on pooled data 
for itemname, filename in dname.items():
    print('working with', filename)
    iname = itemname + '__'
    
    d = EEGDataLoader(filename)
    # load subject specific data 
    s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]            
    d1 = d.load_pooled(s)
    
    #% identify input size (channel x timepoints)
    timelength = d1['xtest'].shape[2]
    chans = d1['xtest'].shape[1]
    input_size = (1, chans, timelength)
    
    #% used to save the results table 
    results = {}        
    table = pd.DataFrame(columns = ['Train_Acc', 'Val_Acc', 'Test_Acc', 'Epoch'])       
    
    # get data 
    dat = get_data(d1, batch_size, image = True, lstm = False, raw = False)   
    dset_loaders = dat['dset_loaders']
    dset_sizes   = dat['dset_sizes']   
            
    # here we define different combination of hyperparameters with varying level of 
    # cnn and lstm layers, and kernel size. 
    if ConvDOWN:            
        params = {'conv_channels': [[1, 16, 8],    # convolutional layers and number of filters in each layer                                                  
                                    [1, 32, 16, 8],
                                    [1, 64, 32, 16, 8],
                                    [1, 128, 64, 32, 16, 8],
                                    [1, 256, 128, 64, 32, 16, 8], 
                                    
                                    [1, 32, 16],                                                       
                                    [1, 64, 32, 16],
                                    [1, 128, 64, 32, 16],
                                    [1, 256, 128, 64, 32, 16],
                                    [1, 256, 256, 128, 64, 32, 16]],                                    
    					
                  'kernel_size':    [[3, 3, 3, 3, 3, 3], # kernel size
                                     [7, 7, 5, 5, 3, 3],
                                     [13, 11, 9, 7, 5, 3]]}    
                  
    else:                                               
        params = {'conv_channels': [[1, 8, 16],  # convolutional layers and number of filters in each layer                                                          
                                    [1, 8, 16, 32],
                                    [1, 8, 16, 32, 64],
                                    [1, 8, 16, 32, 64, 128],
                                    [1, 8, 16, 32, 64, 128, 256],       
                                    [1, 16, 32],                                                       
                                    [1, 16, 32, 64],
                                    [1, 16, 32, 64, 128],
                                    [1, 16, 32, 64, 128, 256],
                                    [1, 16, 32, 64, 128, 256, 512]],      		
        					
                  'kernel_size':    [[3, 3, 3, 3, 3, 3], # kernel size
                                     [3, 3, 5, 5, 7, 7],
                                     [3, 5, 7, 9, 11, 13]]}
                  
    keys = list(params)
    
    for values in itertools.product(*map(params.get, keys)):     
        d = dict(zip(keys, values))
        description = 'C{}_K{}'.format(d['conv_channels'], d['kernel_size'][:len(d['conv_channels'])])    
        print('\n\n##### ' + description + ' #####')
        
        # Define the model
        model = CNN2D(input_size    = input_size,
                      kernel_size   = d['kernel_size'], 
                      conv_channels = d['conv_channels'])      
            
        print("Model architecture >>>", model)
        # optimizer and the loss function definition 
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        
        # move the model to GPU/CPU
        model.to(dev)  
        criterion.to(dev)       
            
        #******** Training loop *********    
        best_model, train_losses, val_losses, train_accs, val_accs , info = train_model(model, dset_loaders, 
                                                                                        dset_sizes, criterion, optimizer,
                                                                                        dev, lr_scheduler=None, num_epochs=num_epochs,                                                                                     
                                                                                        verbose = verbose)    
        #------------------------------------------------
        # here train_model returns the best_model which is saved for a later use below        
        # we could immediately evaluate the best model on the test as
        
        x_test = dat['test_data']['x_test'] 
        y_test = dat['test_data']['y_test'] 
        #************************
        
        preds = best_model(x_test.to(dev)) 
        preds_class = preds.data.max(1)[1]
        
        # accuracy 
        corrects = torch.sum(preds_class == y_test.data.to(dev))     
        test_acc = corrects.cpu().numpy()/x_test.shape[0]
        print("Test Accuracy :", test_acc) 
        
        # save results       
        tab = dict(Train_Acc  = train_accs[info['best_epoch']],
                   Val_Acc    = val_accs[info['best_epoch']],   
                   Test_Acc   = test_acc, 
                   Epoch      = info['best_epoch'] + 1)   
        
        table.loc[description] = tab  
    
        results[description] = dict(train_accs = train_accs, val_accs =  val_accs,                                
                                    ytrain = info['ytrain'], yval= info['yval'])      
        
        fname = iname + 'CNN_POOLED' + description  
        torch.save(best_model.state_dict(), fname) 

    # save all the results in one file 
    result_cnn = dict(table = table, results = results)
    fname2 = iname + "__CNN_POOLED_RESULTS_ALL"         
    
    with open(fname2, 'wb') as fp:
        pickle.dump(result_cnn, fp)
   