# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:03:25 2019
@author: Berdakh
"""
import torch 
import itertools
import pandas as pd 
import pickle 
from nu_data_loader import getTorch, EEGDataLoader
from nu_train_utils import train_model
from nu_models import LSTM_Model
   
dev = torch.device("cpu")

if torch.cuda.is_available():
    print("CUDA available!")
    dev = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
#% Load ERP data
get_data = getTorch.get_data 

# Pooled data 
dname = dict(nu = 'data_allsubjects.pickle',
             epfl = 'EPFLP300.pickle',
             ten = 'TenHealthyData.pickle',
             als = 'ALSdata.pickle')

#tested 
num_epochs = 100
batch_size = 64
verbose = 2  
learning_rate = 1e-3
weight_decay = 1e-4# L2 regularizer parameter    

#%%
for itemname, filename in dname.items():    
    print('::: Working with :::', filename)
    iname = itemname + '_'
       
    dd = EEGDataLoader(filename)
    
    # subject data indicies 
    s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    #% load pooled data 
    d1 = dd.load_pooled(s)
    
    #% get input size (channel x timepoints)
    timelength = d1['xtest'].shape[2]
    channels   = d1['xtest'].shape[1]
    input_size = channels    

   # Pooled Data LSTM Train    
    table = pd.DataFrame(columns = ['Train_Loss', 'Val_Loss', 'Train_Acc', 
                                    'Val_Acc',    'Test_Acc', 'Epoch'])
       
    dat = get_data(d1, batch_size, image = False, lstm = True, raw = False)
    
    params = {'num_layers': [1, 2, 3],
              'hidden_size': [64, 128, 256]}
    
    keys = list(params)
    dset_loaders = dat['dset_loaders']
    dset_sizes   = dat['dset_sizes']    
   
    results = {}    
    # Training Loop 
    for values in itertools.product(*map(params.get, keys)):   
        d = dict(zip(keys, values))
        description = '_L{}_H{}'.format(d['num_layers'], d['hidden_size'])
        print('\n\n##### ' + description + ' #####')
              
        # Define the model
        model = LSTM_Model(input_size  = input_size, 
                           hidden_size = d['hidden_size'], 
                           num_layers  = d['num_layers'], 
                           dropout     = 0.1)           
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.to(dev)  
        criterion.to(dev)               
        #***************** Training loop ********************
        best_model, train_losses, val_losses, train_accs, val_accs, info = train_model(model, 
                                                                                       dset_loaders, 
                                                                                       dset_sizes, criterion, 
                                                                                       optimizer, dev, 
                                                                                       lr_scheduler = None, 
                                                                                       num_epochs = num_epochs, 
                                                                                       verbose = verbose)
        
        # evaluate the best model   
        x_test = dat['test_data']['x_test']   
        y_test = dat['test_data']['y_test'] 
    
        h = best_model.init_hidden(x_test.shape[0])
        preds = best_model(x_test.to(dev),h)    
        preds_class = preds.data.max(1)[1]
        
        corrects = torch.sum(preds_class == y_test.data.to(dev))     
        test_acc = corrects.cpu().numpy()/x_test.shape[0]    
        print("Test Accuracy :", test_acc)
        #------------------------------------------------
        # save results       
        tab = dict(Train_Loss = train_losses[info['best_epoch']], 
                   Val_Loss   = val_losses[info['best_epoch']],
                   Train_Acc  = train_accs[info['best_epoch']],
                   Val_Acc    = val_accs[info['best_epoch']],   
                   Test_Acc   = test_acc, 
                   Epoch      = info['best_epoch'] + 1)         
        table.loc[description] = tab     

        results[description] = dict(train_losses = train_losses, 
                                    val_losses   = val_losses,
                                    train_accs   = train_accs, 
                                    val_accs     = val_accs,                                
                                    ytrain       = info['ytrain'], 
                                    yval         = info['yval'])    
                
        ######################## SAVE THE MODELS ######################
        fname = iname + 'LSTM_POOLED_model_' + description + '_' + str(info['best_acc'])[:4]+ "__" + str(test_acc)[:4]
        torch.save(best_model.state_dict(), fname)                
        print(table)
        
    # save all the results in one file 
    result_lstm = dict(table = table, results = results)
    fname2 = iname + "LSTM_POOLED_RESULTS_ALL"         
    
    with open(fname2, 'wb') as fp:
        pickle.dump(result_lstm, fp)     