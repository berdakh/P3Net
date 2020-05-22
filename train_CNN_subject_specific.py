# %% Subject specific CNN %%
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:03:25 2019
@author: Berdakh

This script can be used to train CNN model on subject specific data.
"""
import torch
import pandas as pd
import pickle
from nu_data_loader import getTorch, EEGDataLoader
from nu_train_utils import train_model
from nu_models import CNN2D

# to get a torch tensor
get_data = getTorch.get_data
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Your GPU device name :', torch.cuda.get_device_name())

# %% dataset information
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
dname = dict(nu='data_allsubjects.pickle',
             epfl='EPFLP300.pickle',
             ten='TenHealthyData.pickle',
             als='ALSdata.pickle')

# %% Hyperparameter settings
num_epochs = 100
batch_size = 64
verbose = 2
learning_rate = 1e-3
weight_decay = 1e-4

# %%
for itemname, filename in dname.items():
    print('working with', filename)
    iname = itemname + '__'

    # Hyperparamters:
    # here for each datatype, a unique model architecture can be set
    # these architectures are defined based on the pooled data
    if filename == 'data_allsubjects.pickle':
        kernel_size = [13, 11]    # kernel size
        # convolutional layers and number of filters in each layer
        conv_chan = [1, 32, 16]

    elif filename == 'EPFLP300.pickle':
        kernel_size = [7, 7]
        conv_chan = [1, 32, 16]

    elif filename == 'TenHealthyData.pickle':
        kernel_size = [7, 7]
        conv_chan = [1, 32, 16]

    elif filename == 'ALSdata.pickle':
        kernel_size = [3, 3, 3]
        conv_chan = [1, 32, 16, 8]

    # EEG data loader
    d = EEGDataLoader(filename)

    # load subject specific data with subject data indicies
    s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    data = d.subject_specific(s)
    print(data[0].keys())

    # % identify input size (channel x timepoints)
    timelength = data[0]['xtrain'].shape[2]
    chans = data[0]['xtrain'].shape[1]
    input_size = (1, chans, timelength)

    datum = {}
    # get torch data loaders for each subject data
    for ii in range(len(data)):
        datum[ii] = get_data(data[ii], batch_size,
                             image=True, lstm=False, raw=False)

    # used for storing results later
    results = {}
    table = pd.DataFrame(columns=['Train_Loss', 'Val_Loss', 'Train_Acc',
                                  'Val_Acc', 'Test_Acc', 'Epoch'])

    # for each subject in a given data type perform model selection
    for subjectIndex in datum:

        # Define the model
        model = CNN2D(input_size=input_size,
                      kernel_size=kernel_size,
                      conv_channels=conv_chan)
        print(model)
        # define the optimizer and the loss function
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        # move to GPU/CPU
        model.to(dev)
        criterion.to(dev)

        # get data loaders for each subject
        dset_loaders = datum[subjectIndex]['dset_loaders']
        dset_sizes = datum[subjectIndex]['dset_sizes']

        print('::: processing subject :::', subjectIndex)
        # ***************** Training loop ********************
        best_model, train_losses, val_losses, train_accs, val_accs, info = train_model(model, dset_loaders,
                                                                                       dset_sizes,
                                                                                       criterion, optimizer,
                                                                                       dev, lr_scheduler=None,
                                                                                       num_epochs=num_epochs,
                                                                                       verbose=verbose)
        # ------------------------------------------------
        # here train_model returns the best_model which is saved for a later use below
        # we could immediately evaluate the best model on the test as
        x_test = datum[subjectIndex]['test_data']['x_test']
        y_test = datum[subjectIndex]['test_data']['y_test']

        preds = best_model(x_test.to(dev))
        preds_class = preds.data.max(1)[1]

        corrects = torch.sum(preds_class == y_test.data.to(dev))
        test_acc = corrects.cpu().numpy()/x_test.shape[0]
        print("Test Accuracy :", test_acc)

        # save results
        tab = dict(Train_Loss=train_losses[info['best_epoch']],
                   Val_Loss=val_losses[info['best_epoch']],
                   Train_Acc=train_accs[info['best_epoch']],
                   Val_Acc=val_accs[info['best_epoch']],
                   Test_Acc=test_acc,
                   Epoch=info['best_epoch']+1)

        table.loc[subjectIndex+1] = tab

        # save the best model
        modelname = iname + 'S' + str(subjectIndex) + '_CNN'+".model"
        torch.save(best_model.state_dict(), modelname)

        results[subjectIndex] = dict(train_losses=train_losses, val_losses=val_losses,
                                     train_accs=train_accs,     val_accs=val_accs,
                                     ytrain=info['ytrain'],     yval=info['yval'])
        resultat = dict(table=table, results=results)

        fname2 = iname + "__CNN_subspe_results"
    with open(fname2, 'wb') as fp:
        pickle.dump(resultat, fp)
