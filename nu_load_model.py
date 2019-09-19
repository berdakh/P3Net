# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:27:47 2019
@author: berdakh.abibullaev
"""
import pickle
import torch
from nu_models import CNN
model = CNN() 

#%% 
def loadfile(filename):    
    f = open(filename,'rb')
    data = pickle.load(f)
    f.close()
    return data     
#%%    
filename = 'CNN_model_94.3'
d = torch.load(filename)

# insert the weights
model.state_dict(d)    

