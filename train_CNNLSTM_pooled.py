import torch 
import pandas as pd 
import pickle 

import itertools
from nu_data_loader import getTorch, EEGDataLoader
from nu_train_utils import train_model
from nu_models import CNN2DEncoder, CNNLSTM
    
dev = torch.device("cpu")
if torch.cuda.is_available():
    print("CUDA available!")
    dev = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Load ERP data
get_data = getTorch.get_data 
#%
dname = dict(nu   = 'data_allsubjects.pickle',
             epfl = 'EPFLP300.pickle',
             ten  = 'TenHealthyData.pickle',
             als  = 'ALSdata.pickle')

# tested 
batch_size = 64      
num_epochs = 100
verbose = 2 
learning_rate = 1e-3
weight_decay = 1e-4 

#%%
for itemname, filename in dname.items():
    print('working with', filename)
    iname = itemname + '__'

    d = EEGDataLoader(filename)
    
    # subject data indicies 
    s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    table = pd.DataFrame(columns = ['Train_Loss', 'Val_Loss', 'Train_Acc', 
                                    'Val_Acc',    'Test_Acc', 'Epoch'])
    
    d1 = d.load_pooled(s)
    
     #% identify input size (channel x timepoints)
    timelength = d1['xtest'].shape[2]
    chans = d1['xtest'].shape[1]
    input_size = (10, 1, chans, timelength)    
               
    ConvDOWN = True   # change this option 
    
    if ConvDOWN:            
        params = {'conv_channels': [[1, 16, 8],                                                       
                                    [1, 32, 16, 8],
                                    [1, 64, 32, 16, 8],
                                    [1, 128, 64, 32, 16, 8],
                                    [1, 256, 128, 64, 32, 16, 8], 
                                    
                                    [1, 32, 16],                                                       
                                    [1, 64, 32, 16],
                                    [1, 128, 64, 32, 16],
                                    [1, 256, 128, 64, 32, 16],
                                    [1, 256, 256, 128, 64, 32, 16]],     

                  'kernel_size':    [[3, 3, 3, 3, 3, 3],
                                     [7, 7, 5, 5, 3, 3],
                                     [13, 11, 9, 7, 5, 3]],
                                     
                  'num_layers':      [1, 2],
                  'hidden_size':     [64, 128]}       
    else:                                               
        params = {'conv_channels': [[1, 8, 16],                                                       
                                    [1, 8, 16, 32],
                                    [1, 8, 16, 32, 64],
                                    [1, 8, 16, 32, 64, 128],
                                    [1, 8, 16, 32, 64, 128, 256],   
                                    
                                    [1, 16, 32],                                                       
                                    [1, 16, 32, 64],
                                    [1, 16, 32, 64, 128],
                                    [1, 16, 32, 64, 128, 256],
                                    [1, 16, 32, 64, 128, 256, 512]],      		
        					
                  'kernel_size':    [[3, 3, 3, 3, 3, 3],
                                     [3, 3, 5, 5, 7, 7],
                                     [3, 5, 7, 9, 11, 13]],
                                     
                  'num_layers':      [1, 2],
                  'hidden_size':     [64, 128]}                                          
                  
    keys = list(params)   
    results = {}           
    ii = 0        
    # Train Loop 
    for values in itertools.product(*map(params.get, keys)):   
        d = dict(zip(keys, values))
        
        kernel_size = d['kernel_size'][:len(d['conv_channels'])-1]
        
        description = '_L{}_H{}_C{}_K{}'.format(d['num_layers'], d['hidden_size'],
                         d['conv_channels'], kernel_size)
        print('\n\n##### ' + description + ' #####')
        ii += 1
        print(ii)
      # define the encoder model   
        encoder = CNN2DEncoder(kernel_size = kernel_size, conv_channels = d['conv_channels'])      

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
        
        print('********** Model Architecture ****************')
        print(model)
        
        dat = get_data(d1, batch_size, image = True, lstm = False, raw = False)        
        dset_loaders = dat['dset_loaders']
        dset_sizes   = dat['dset_sizes']    
        
        m = 0
        results = {}        
        #% Train Loop     
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        
        model.to(dev)  
        criterion.to(dev)          
        
       #******** Training loop *********    
        best_model, train_losses, val_losses, train_accs, val_accs , info = train_model(model, dset_loaders, dset_sizes, 
                                                                                        criterion, optimizer,
                                                                                        dev, lr_scheduler=None, 
                                                                                        num_epochs=num_epochs, 
                                                                                        verbose = verbose)    
         
        # evaluate the best model   
        x_test = dat['test_data']['x_test'] 
        y_test = dat['test_data']['y_test'] 
        
        h=best_model.init_hidden(x_test.shape[0])
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
        results[description] = dict(train_accs = train_accs, val_accs =  val_accs,                                
                                    ytrain = info['ytrain'], yval= info['yval'])      
        
        fname = iname + 'CNNLSTMPOOLED' + description + '_' + str(info['best_acc'])[:4]+ "__" + str(test_acc)
        torch.save(best_model.state_dict(), fname)        
        print(table)
        
    # save all the results 
    result_cnnlstm = dict(table = table, results = results)
    fname2 = iname + "__CNNLSTMPOOLED_RESULTS_ALL_3RD"         
    
    with open(fname2, 'wb') as fp:
        pickle.dump(result_cnnlstm, fp)
        