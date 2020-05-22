'''
This script evaluates the performance of the LRR classifier is evaluated on ERP data preprocessed using three different approaches: 

- Xdawn spatial filtering technique for EEG data preprocessing \cite{rivet2009xdawn, rivet2011theoretical}. Xdawn is a spatial filter developed for improving the signal to signal + noise ratio (SSNR) of event-related potentials (ERP). Originally, Xdawn has developed specifically for P300-ERPs based BCIs to enhance the target response concerning the non-target response. It has also been shown powerfully in recent studies as a part of best practice for ERP studies \cite{Cecotti2017};

- Riemannian geometry method for preprocessing of EEG data \cite{barachant2011multiclass, barachant2013classification}. Recent research demonstrates the state-of-the-art performances of Riemannian geometry-based methods on different BCI problems \cite{yger2016riemannian, lotte2018review}, along with deep-learning methods. 


'''
import os 
import pandas as pd 
from sklearn.linear_model import LinearRegression as LRR
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
#%%
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import XdawnCovariances, Xdawn
# %% Import moabb 
import moabb
from moabb.paradigms import P300
from moabb.datasets import BNCI2015003
# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
moabb.set_log_level('info')

# ____________________________________________________________________________
# This is an auxiliary transformer that allows one to vectorize data
# structures in a pipeline For instance, in the case of a X with dimensions
# Nt x Nc x Ns, one might be interested in a new data structure with
# dimensions Nt x (Nc.Ns)
class Vectorizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """
        return np.reshape(X, (X.shape[0], -1))

# %%
# ____________________________________________________________________________
# Create pipelines
# ----------------
# Pipelines must be a dict of sklearn pipeline transformer.
pipelines = {}
# we have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {'Target': 1, 'NonTarget': 0}

# %%  
# from sklearn.preprocessing import StandardScaler
                              
pipelines['RG + LRR'] = make_pipeline(
    XdawnCovariances(
        nfilter=2,
        classes=[labels_dict['Target']],
        estimator='lwf',
        xdawn_estimator='lwf'),    
    TangentSpace(),
    LRR())

# %%
pipelines['Xdawn + LRR'] = make_pipeline(Xdawn(nfilter=2, estimator='lwf'),
                                         Vectorizer(), LRR())

#%%
pipelines['LRR'] = make_pipeline(Vectorizer(), LRR())

# ____________________________________________________________________________
# Evaluation
# %%
paradigm = P300(resample=128)
dataset  = BNCI2015003()

#number of subjects 
print(dataset.subject_list)
#%%
dataset.subject_list = dataset.subject_list[:]
datasets = [dataset]

# %%
from moabb.evaluations import WithinSessionEvaluation

overwrite = True  # set to True if we want to overwrite cached results
evaluation = WithinSessionEvaluation(paradigm=paradigm,
                                    datasets=datasets,
                                    suffix='examples',
                                    overwrite=overwrite)

#%%
results = evaluation.process(pipelines)
print(results)
