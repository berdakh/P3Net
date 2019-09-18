#%%
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

# to get a torch tensor 
get_data = getTorch.get_data 
dev = torch.device("cpu")

if torch.cuda.is_available():    
    dev = torch.device("cuda") 
    print('Your GPU device name :', torch.cuda.get_device_name())    

dname = dict(nu = 'data_allsubjects.pickle',
             epfl = 'EPFLP300.pickle',
             ten = 'TenHealthyData.pickle',
             als = 'ALSdata.pickle')

#tested 
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-4 
num_epochs = 100
verbose = 1  

#%%       
for itemname, filename in dname.items():
    print('working with', filename)
    iname = itemname + '__'
    
    if filename =='data_allsubjects.pickle':
        num_layers = 3
        hidden_size = 64
    elif filename =='EPFLP300.pickle':
        num_layers = 3
        hidden_size = 64
    elif filename =='TenHealthyData.pickle':
        num_layers = 3
        hidden_size = 128
    elif filename =='ALSdata.pickle':
        num_layers = 3
        hidden_size = 128
        
    d = EEGDataLoader(filename)   
    # subject data indicies 
    s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]        
    data = d.subject_specific(s)
    print(data[0].keys())
    
    
    #% identify input size (channel x timepoints)
    timelength = data[0]['xtrain'].shape[2]
    chans      = data[0]['xtrain'].shape[1]
    input_size = (1, chans, timelength)
    
    datum = {}
    
    for ii in range(len(data)):
      datum[ii] = get_data(data[ii], batch_size = batch_size, 
                           image = False, lstm = True, raw = False)
    
    results, models = {}, {}       
    table = pd.DataFrame(columns = ['Train_Loss', 'Val_Loss', 'Train_Acc', 
                                        'Val_Acc', 'Test_Acc', 'Epoch'])
        
    for subjectIndex in datum:            
        dset_loaders = datum[subjectIndex]['dset_loaders']
        dset_sizes   = datum[subjectIndex]['dset_sizes']    
        
        # instantiate the model 
        model = LSTM_Model(chans, hidden_size = hidden_size, num_layers = num_layers, dropout = 0.1)                   
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
    
        model.to(dev)
        criterion.to(dev)
          
        #***************** Training loop ********************
        best_model, train_losses, val_losses, train_accs, val_accs , info = train_model(model, 
                                                                                       dset_loaders, 
                                                                                       dset_sizes, 
                                                                                       criterion,optimizer, 
                                                                                       dev, lr_scheduler = None, 
                                                                                       num_epochs=num_epochs, 
                                                                                       verbose = verbose)
        #------------------------------------------------
        # evaluate the best model   
        x_test = datum[subjectIndex]['test_data']['x_test'] 
        y_test = datum[subjectIndex]['test_data']['y_test'] 
         
        h = best_model.init_hidden(x_test.shape[0])
        preds = best_model(x_test.to(dev),h)    
        preds_class = preds.data.max(1)[1]
    
        corrects = torch.sum(preds_class == y_test.data.to(dev))     
        test_acc = corrects.cpu().numpy()/x_test.shape[0] 
        print("Test accuracy", test_acc)
          
        #------------------------------      
        tab = dict(Train_Loss = train_losses[info['best_epoch']], 
                   Val_Loss   = val_losses[info['best_epoch']],
                   Train_Acc  = train_accs[info['best_epoch']],
                   Val_Acc    = val_accs[info['best_epoch']],   
                   Test_Acc   = test_acc, 
                   Epoch = info['best_epoch'] +1 )  
          
        table.loc[subjectIndex] = tab          
        results[subjectIndex] = dict(train_losses = train_losses, val_losses = val_losses,
                                    train_accs = train_accs,     val_accs =  val_accs,                                
                                    ytrain = info['ytrain'],     yval= info['yval'])      
          
        # save models separately   
        fname = iname + 'S'+ str(subjectIndex) + '_LSTM_model_'+ str(info['best_acc'])[:4] + "__" + str(test_acc)     
        torch.save(best_model.state_dict(), fname) 
            
        print('::: saving subject {} ::: \n {}'.format(subjectIndex, table))         
        result_lstm_subspe = dict(table = table, results = results)          
          
    fname = iname + '__S'+str(subjectIndex)  + '_LSTM_subspe_results'
    
    with open(fname, 'wb') as fp:
        pickle.dump(result_lstm_subspe, fp)                
