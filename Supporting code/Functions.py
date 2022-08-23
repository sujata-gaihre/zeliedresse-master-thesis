#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:24:02 2022

@author: zeliedresse
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_curve, auc, roc_auc_score

def depth_plot(parameters, X_train, y_train, X_val, y_val):    
    """
    Takes list of possible parameters for max_depth in the random forest classifier
    and returns plot with the corresponding AUC scores for models with the default parameters.
    """

    maxdepth_list = []
    for i in parameters:
        rf = RandomForestClassifier(max_depth = i, random_state = 8, n_jobs = -1,
                                    n_estimators = 100, criterion = "gini")
        rf.fit(X_train, y_train)
        rf_prob = rf.predict_proba(X_val)
        rf_prob = rf_prob[:,1]
        fpr_rf, tpr_rf, threshold = roc_curve(y_val, rf_prob)
        maxdepth_list.append(auc(fpr_rf, tpr_rf))
    
    fig = plt.figure(0)
    plt.plot(parameters, maxdepth_list)
    
    return(fig)        

#%%
def mtry_plot(parameters, X_train, y_train, X_val, y_val, max_depth):
    """
    Takes list of possible parameters for mtry in the random forest classifier
    and returns plot with the corresponding AUC scores for models with the default parameters
    and the chosen max_depth parameter
    """
    mtry_list = []
    for i in parameters:
        rf = RandomForestClassifier(max_depth = max_depth, random_state = 8, n_jobs = -1,
                                    n_estimators = 100, criterion = "gini",
                                    max_features = i)
        rf.fit(X_train, y_train)
        rf_prob = rf.predict_proba(X_val)
        rf_prob = rf_prob[:,1]
        fpr_rf, tpr_rf, threshold = roc_curve(y_val, rf_prob)
        mtry_list.append(auc(fpr_rf, tpr_rf))
    
    fig = plt.figure(0)
    plt.plot(parameters, mtry_list)
    return(fig)       
 
#%%
def plot_roc(model, X_val, y_val):
    """
    Takes trained model and validation data and returns plot figure, 
    area under the ROC curve, true positive rates, false positives rate 
    with the corresponding thresholds.
    """
    
    prob = model.predict_proba(X_val)
    prob = prob[:,1]

    fpr, tpr, threshold = roc_curve(y_val, prob)
    roc_auc = roc_auc_score(y_val, prob)
    
    fig = plt.figure(0)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    return(fig, roc_auc, tpr, fpr, threshold)


#%%
def get_auc(model, X_val, y_val):
    """
    Takes a trained model and validation data and returns the validation area under the ROC curve. 
    """
    prob = model.predict_proba(X_val)
    prob = prob[:,1]

    roc_auc = roc_auc_score(y_val, prob)
    
    return(roc_auc)

#%%
def tpr_10fpr(tpr, fpr):
    """
    Takes the true positive rates and the false positive rates and returns the TPR
    where the FPR is equal to 10%.    
    """
    index = np.argmin(abs(fpr - 0.1))
    rate10perc = tpr[index]
    
    return(rate10perc)
