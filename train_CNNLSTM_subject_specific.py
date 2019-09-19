#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 21:47:08 2019
@author: berdakh

This script can be used to train CNNLSTM model on subject specific data.

"""
import torch 
import pandas as pd 
import pickle 

from nu_data_loader import getTorch, EEGDataLoader
from nu_train_utils import train_model
from nu_models import CNN2DEncoder, CNNLSTM
 

dev = torch.device("cpu")
if torch.cuda.is_available():
    print("CUDA available!")
    dev = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


get_data = getTorch.get_data 


#%% dataset information
'''
dname is a dictionary containing dataset names to be loaded from
the current directory

The following files represent the ERP datasets referred in the paper as:

    NU data   = 'data_allsubjects.pickle',
    EPFL data = 'EPFLP300.pickle'
    BNCI data ='TenHealthyData.pickle'
    ALS data  ='ALSdata.pickle'
'''

# Load ERP data
dname = dict(nu   = 'data_allsubjects.pickle',
             epfl = 'EPFLP300.pickle',
             ten  = 'TenHealthyData.pickle',
             als  = 'ALSdata.pickle')

#%% Hyperparameter settings 

num_epochs = 100
batch_size = 64
verbose = 2
learning_rate = 1e-3
weight_decay = 1e-4     

#%%
for itemname, filename in dname.items():
    print('working with', filename)
    iname = itemname + '__'   
    
    # Hyperparamters:    
    # here for each datatype, a unique model architecture can be set
    # these architectures are defined based on the pooled data     
    
    if filename =='data_allsubjects.pickle':
        d  =  {'conv_channels': [1, 256, 128, 64, 32, 16, 8], # convolutional layers and number of filters in each layer 
               'kernel_size':   [3, 3, 3, 3, 3, 3], # kernel size 
               'num_layers':    1, # number of LSTM layers
               'hidden_size':   128} # number of neurons in each LSTM layer 

    elif filename =='EPFLP300.pickle':
        d  =  {'conv_channels': [1, 32, 16, 8],
               'kernel_size':   [13, 11, 9],
               'num_layers':    1,
               'hidden_size':   64}
        
    elif filename =='TenHealthyData.pickle':
        d  =  {'conv_channels': [1, 64, 32, 16, 8],
               'kernel_size':   [13, 11, 9, 7],
               'num_layers':    2,
               'hidden_size':   64}

    elif filename =='ALSdata.pickle':        
        d  =  {'conv_channels': [1, 128, 64, 32, 16, 8],
               'kernel_size':   [13, 11, 9, 7, 5],
               'num_layers':    1,
               'hidden_size':   64}
  
    
    # EEG data loader 
    dd = EEGDataLoader(filename)
    
    # load subject specific data with subject data indicies  
    s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   
    data = dd.subject_specific(s)
    print(data[0].keys())
        
    #% identify input size (channel x timepoints)
    timelength = data[0]['xtrain'].shape[2]
    chans      = data[0]['xtrain'].shape[1]
 
    datum = {}
    # get torch data loaders for each subject data 
    for ii in range(len(data)):
      datum[ii] = get_data(data[ii], batch_size, image = True, lstm = False, raw = False)
   
    # used for storing results later 
    results, models = {}, {}  
    table = pd.DataFrame(columns = ['Train_Loss', 'Val_Loss', 'Train_Acc', 
                                        'Val_Acc', 'Test_Acc', 'Epoch'])
    
    # for each subject in a given data type perform model selection
    for subjectIndex in datum:    
        dset_loaders = datum[subjectIndex]['dset_loaders']
        dset_sizes   = datum[subjectIndex]['dset_sizes']      
        
        input_size = datum[subjectIndex]['test_data']['x_test'].shape

        # define the encoder model                  
        encoder = CNN2DEncoder(kernel_size = d['kernel_size'], 
                               conv_channels = d['conv_channels'])      
        # we need to pass a sample data to get the output dimensionality of CNN 
        with torch.no_grad():
            x = torch.randn(input_size)
            outdim = encoder(x)
            batch_size, chans, H, W = outdim.size()
            
        # define the encoder-decoder model 
        model = CNNLSTM(input_size = H*W, 
                        cnn = encoder,
                        hidden_size = d['hidden_size'], 
                        num_layers  = d['num_layers'], 
                        batch_size  = 64,
                        dropout     = 0.2)      
        
        # define the optimizer and the loss function 
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        
        # move to GPU/CPU
        model.to(dev)  
        criterion.to(dev)           
        
        # the training loop
        best_model, train_losses, val_losses, train_accs, val_accs, info = train_model(model, dset_loaders, 
                                                                                       dset_sizes, criterion, 
                                                                                       optimizer, dev, 
                                                                                       lr_scheduler = None, 
                                                                                       num_epochs = num_epochs, 
                                                                                       verbose = verbose)    
         
        # here train_model returns the best_model which is saved for a later use below        
        # we could immediately evaluate the best model on the test as
        x_test = datum[subjectIndex]['test_data']['x_test'] 
        y_test = datum[subjectIndex]['test_data']['y_test'] 
        
        h=best_model.init_hidden(x_test.shape[0])
        preds = best_model(x_test.to(dev),h)    
        preds_class = preds.data.max(1)[1]
        
        # accuracy metric 
        corrects = torch.sum(preds_class == y_test.data.to(dev))     
        test_acc = corrects.cpu().numpy()/x_test.shape[0]    
        print("Test Accuracy :", test_acc)    
        
        # save results       
        tab = dict(Train_Loss = train_losses[info['best_epoch']], 
                   Val_Loss   = val_losses[info['best_epoch']],
                   Train_Acc  = train_accs[info['best_epoch']],
                   Val_Acc    = val_accs[info['best_epoch']],   
                   Test_Acc   = test_acc, 
                   Epoch      = info['best_epoch'] + 1) 
        
        subjectIndexx = subjectIndex + 1
        table.loc[subjectIndexx] = tab
        
        results[subjectIndexx] = dict(train_losses = train_losses, val_losses = val_losses,
                                    train_accs = train_accs,     val_accs =  val_accs,                                
                                    ytrain = info['ytrain'],     yval= info['yval'])      
          
        # save models   
        fname = iname + 'S'+ str(subjectIndexx) + '_CNNLSTM_model_'
        torch.save(best_model.state_dict(), fname) 
         
        print('::: saving subject {} ::: \n {}'.format(subjectIndexx, table))         
        result_lstm_subspe = dict(table = table, results = results)          
          
    fname = iname + '__S'+str(subjectIndexx)  + '_CNNLSTM_subspe_results'
    
    with open(fname, 'wb') as fp:
        pickle.dump(result_lstm_subspe, fp)                
            