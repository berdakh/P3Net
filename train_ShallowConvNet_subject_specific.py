"""
@author: berdakh

This script can be used to train subject-speficic ShallowConvNet model. Note that we used the ShallowConvNet implementation 
provided via the brain decode toolbox available at https://robintibor.github.io/braindecode/.

You can install the library via >>> pip install braindecode
"""

from nu_train_utils import train_model
import pickle
import pandas as pd
import torch
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from nu_data_loader import getTorch, EEGDataLoader
import sys
import logging
import importlib
importlib.reload(logging)
log = logging.getLogger()
log.setLevel('INFO')

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)

get_data = getTorch.get_dataEEGnet

# import the model
############################################################
############################################################
Model = ShallowFBCSPNet

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Your GPU device name :', torch.cuda.get_device_name())

# %%
'''
dname is a dictionary containing dataset names to be loaded from
the current directory

The following files represent the ERP datasets referred in the paper as:

    NU data   = 'data_allsubjects.pickle',
    EPFL data = 'EPFLP300.pickle'
    BNCI data ='TenHealthyData.pickle'
    ALS data  ='ALSdata.pickle'
'''

dname = dict(nu='data_allsubjects.pickle',
             epfl='EPFLP300.pickle',
             ten='TenHealthyData.pickle',
             als='ALSdata.pickle')

# %% Hyperparameter settings
num_epochs = 100
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 64
verbose = 2
n_classes = 2

# %% The main loop starts here
# for each dataset in dname train the model on subject specific data
for itemname, filename in dname.items():
    print('working with', filename)
    iname = itemname + '__'

    d = EEGDataLoader(filename)
    # subject data indicies
    s = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    data = d.subject_specific(s)
    print(data[0].keys())

    # % identify input size (channel x timepoints)
    timelength = data[0]['xtrain'].shape[2]
    chans = data[0]['xtrain'].shape[1]

    datum = {}
    # get torch data loaders
    for ii in range(len(data)):
        datum[ii] = get_data(data[ii], batch_size, lstm=False, image=True)

    # used for storing the results
    results, models = {}, {}
    table = pd.DataFrame(columns=['Train_Loss', 'Val_Loss', 'Train_Acc',
                                  'Val_Acc', 'Test_Acc', 'Epoch'])

    # for each subject data performs training the shallowConvNet
    for subjectIndex in datum:
        dset_loaders = datum[subjectIndex]['dset_loaders']
        dset_sizes = datum[subjectIndex]['dset_sizes']

        # final_conv_length = auto ensures we only get a single output in the time dimension
        model = Model(in_chans=chans,
                      n_classes=n_classes,
                      input_time_length=timelength,
                      n_filters_time=15,
                      filter_time_length=25,
                      n_filters_spat=20,
                      pool_time_length=30,
                      pool_time_stride=10,
                      final_conv_length=3,
                      pool_mode="mean",
                      split_first_layer=True,
                      batch_norm=True,
                      batch_norm_alpha=0.1,
                      drop_prob=0.5)

        # create the network
        model = model.create_network()

        # optimizer and the loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        # move to GPU/CPU
        model.to(dev)
        criterion.to(dev)

        # ***************** Training loop ********************
        best_model, train_losses, val_losses, train_accs, val_accs, info = train_model(model,
                                                                                       dset_loaders,
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

        # accuracy
        corrects = torch.sum(preds_class == y_test.data.to(dev))
        test_acc = corrects.cpu().numpy()/x_test.shape[0]
        print("Test accuracy", test_acc)

        # save results
        tab = dict(Train_Loss=train_losses[info['best_epoch']],
                   Val_Loss=val_losses[info['best_epoch']],
                   Train_Acc=train_accs[info['best_epoch']],
                   Val_Acc=val_accs[info['best_epoch']],
                   Test_Acc=test_acc,
                   Epoch=info['best_epoch'] + 1)

        table.loc[subjectIndex] = tab
        results[subjectIndex] = dict(train_losses=train_losses, val_losses=val_losses,
                                     train_accs=train_accs,     val_accs=val_accs,
                                     ytrain=info['ytrain'],     yval=info['yval'])

        print(table)
        # save models separately
        fname = iname + 'S' + str(subjectIndex) + '_FBCSP_model_'
        torch.save(best_model.state_dict(), fname)

        print('::: saving subject {} ::: \n {}'.format(subjectIndex, table))
        result_lstm_subspe = dict(table=table, results=results)

    fname = iname + '__S'+str(subjectIndex) + '_FBCSP_subspe_results'
    with open(fname, 'wb') as fp:
        pickle.dump(result_lstm_subspe, fp)
