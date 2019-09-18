#!/usr/bin/env python
# coding: utf-8
# tested works 
#%%
import logging
import importlib
importlib.reload(logging) # see https://stackoverflow.com/a/21475297/1469195
log = logging.getLogger()
log.setLevel('INFO')
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)
import torch 
import pandas as pd 
import pickle 

from nu_data_loader import getTorch, EEGDataLoader
from nu_train_utils import train_model   
get_data = getTorch.get_dataEEGnet 

from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
 
Model = ShallowFBCSPNet

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Your GPU device name :', torch.cuda.get_device_name())    

# final_conv_length = auto ensures we only get a single output in the time dimension
model = Model(in_chans = 2, 
              n_classes = 2,
              input_time_length = 76,
              n_filters_time = 15,
              filter_time_length = 25,
              n_filters_spat = 20,
              pool_time_length = 30,
              pool_time_stride = 10,
              final_conv_length= 3,
              pool_mode = "mean",
              split_first_layer=True,
              batch_norm = True,
              batch_norm_alpha = 0.1,
              drop_prob = 0.1) 

model = model.create_network()

dname = dict(nu = 'data_allsubjects.pickle',
             epfl = 'EPFLP300.pickle',
             ten = 'TenHealthyData.pickle',
             als = 'ALSdata.pickle')

# tested 
num_epochs = 100 
learning_rate = 1e-3
weight_decay = 1e-4  
batch_size = 64 
verbose = 2
n_classes = 2       

#%%
for itemname, filename in dname.items():
    print('working with', filename)
    iname = itemname + '__'    
    
    # just try nu data below 
    if filename =='data_allsubjects.pickle':
        normalize = False
    elif filename =='EPFLP300.pickle':
        normalize = False
    elif filename =='TenHealthyData.pickle':
        normalize = True
    elif filename =='ALSdata.pickle':
        normalize = True    
        
    results = {}        
    d = EEGDataLoader(filename)
    
    # subject data indicies 
    s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]         
    d1 = d.load_pooled(s)
        
    #% identify input size (channel x timepoints)
    timelength = d1['xtest'].shape[2]
    chans = d1['xtest'].shape[1]        
        
    # dict_keys(['xtrain', 'xvalid', 'xtest', 'ytrain', 'yvalid'])
    dat = get_data(d1, batch_size, lstm = False, image = True)
        
    dset_loaders = dat['dset_loaders']
    dset_sizes   = dat['dset_sizes']    
    
    table = pd.DataFrame(columns = ['Train_Acc', 'Val_Acc', 'Test_Acc', 'Epoch'])
     
    # final_conv_length = auto ensures we only get a single output in the time dimension
    model = Model(in_chans = chans, 
                  n_classes = n_classes,
                  input_time_length = timelength,
                  n_filters_time = 15,
                  filter_time_length = 25,
                  n_filters_spat = 20,
                  pool_time_length = 30,
                  pool_time_stride = 10,
                  final_conv_length= 3,
                  pool_mode = "mean",
                  split_first_layer=True,
                  batch_norm = True,
                  batch_norm_alpha = 0.1,
                  drop_prob = 0.1) 
    
    model = model.create_network()

    print("Model architecture >>>", model)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, 
                                 weight_decay = weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.to(dev)  
    criterion.to(dev)       
            
    #******** Training loop *********    
    best_model, train_losses, val_losses, train_accs, val_accs , info = train_model(model, dset_loaders, 
                                                                                    dset_sizes, 
                                                                                    criterion, optimizer,
                                                                                    dev, lr_scheduler=None,
                                                                                    num_epochs=num_epochs,                                                                                     
                                                                                    verbose = verbose)    
    #%******* TEST ***********
    x_test = dat['test_data']['x_test'] 
    y_test = dat['test_data']['y_test'] 
    #************************
    
    preds = best_model(x_test.to(dev)) 
    preds_class = preds.data.max(1)[1]
    
    corrects = torch.sum(preds_class == y_test.data.to(dev))     
    test_acc = corrects.cpu().numpy()/x_test.shape[0]
    print("Test Accuracy :", test_acc)    
    # save results       
    tab = dict(Train_Acc  = train_accs[info['best_epoch']],
               Val_Acc    = val_accs[info['best_epoch']],   
               Test_Acc   = test_acc, 
               Epoch = info['best_epoch'] + 1)  
    
    description = 'FBCSP_pooled'
    table.loc[description] = tab  
    
    results[description] = dict(train_accs = train_accs, val_accs =  val_accs,                              
                                ytrain = info['ytrain'], yval= info['yval'])      
    
    fname = iname + 'FBCSP_pooled' + description + '_' + str(info['best_acc'])[:4]+ "__" + str(test_acc)
    torch.save(best_model.state_dict(), fname) 
    
    # save all the results in one file 
    result_cnn = dict(table = table, results = results)
    fname2 = iname + "__FBCSP_POOLED_RESULTS_ALL"         
    
    with open(fname2, 'wb') as fp:
        pickle.dump(result_cnn, fp) 


      
