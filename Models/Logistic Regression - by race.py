#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:35:18 2022

@author: zeliedresse

Developing separate logistic regression models for black versus non-black subgroups.
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Import dataset
df = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/data_v2_spon.pkl")

#%% Black separately from rest
from sklearn.model_selection import train_test_split
df_black = df[df["MRACE6"] == 2]
df_nonblack = df[df["MRACE6"] != 2]

X_nb = df_nonblack.drop(columns = ["OEGest_R10", "preterm", "ILLB_R11", "PWgt_R", "BMI", "ME_ROUT", "ME_TRIAL", "LD_INDL","PRECARE"])
y_nb = df_nonblack["preterm"]

X_bl = df_black.drop(columns = ["OEGest_R10", "preterm", "ILLB_R11", "PWgt_R", "BMI", "ME_ROUT", "ME_TRIAL", "LD_INDL", "MRACE6","PRECARE"])
y_bl = df_black["preterm"]

X_train_nb, X_2_nb, y_train_nb, y_2_nb = train_test_split(X_nb, y_nb, test_size = 0.4, random_state = 8)
X_val_nb, X_test_nb, y_val_nb, y_test_nb = train_test_split(X_2_nb, y_2_nb, test_size = 0.5, random_state = 8)

X_train_bl, X_2_bl, y_train_bl, y_2_bl = train_test_split(X_bl, y_bl, test_size = 0.4, random_state = 8)
X_val_bl, X_test_bl, y_val_bl, y_test_bl = train_test_split(X_2_bl, y_2_bl, test_size = 0.5, random_state = 8)

del X_2_nb, y_2_nb, X_2_bl, y_2_bl

#%%
# Continue with standardization + random undersampling
from sklearn.preprocessing import StandardScaler
scaler_train = StandardScaler().fit(X_train_nb)
X_train_sc_nb = pd.DataFrame(scaler_train.transform(X_train_nb),
                          columns =scaler_train.get_feature_names_out())

scaler_val = StandardScaler().fit(X_val_nb)
X_val_sc_nb = pd.DataFrame(scaler_val.transform(X_val_nb), 
                        columns = scaler_val.get_feature_names_out())

scaler_test = StandardScaler().fit(X_test_nb)
X_test_sc_nb = pd.DataFrame(scaler_test.transform(X_test_nb),
                         columns = scaler_test.get_feature_names_out())

scaler_train = StandardScaler().fit(X_train_bl)
X_train_sc_bl = pd.DataFrame(scaler_train.transform(X_train_bl),
                          columns =scaler_train.get_feature_names_out())

scaler_val = StandardScaler().fit(X_val_bl)
X_val_sc_bl = pd.DataFrame(scaler_val.transform(X_val_bl), 
                        columns = scaler_val.get_feature_names_out())

scaler_test = StandardScaler().fit(X_test_bl)
X_test_sc_bl = pd.DataFrame(scaler_test.transform(X_test_bl),
                         columns = scaler_test.get_feature_names_out())

#%%
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state = 8)

X_RUS_nb, y_RUS_nb = rus.fit_resample(X_train_sc_nb, y_train_nb)
X_RUS_bl, y_RUS_bl = rus.fit_resample(X_train_sc_bl, y_train_bl)

#%%
from sklearn.linear_model import LogisticRegression
from functions import get_auc, plot_roc, tpr_10fpr

logistic_nb = LogisticRegression()
logistic_nb.fit(X_RUS_nb.values, y_RUS_nb.values)

auc_nb = get_auc(logistic_nb, X_val_sc_nb, y_val_nb)
#%% evaluation on test data
plot1, auc_test_nb, tpr_nb, fpr_nb, threshold_nb = plot_roc(logistic_nb, X_test_sc_nb,
                                                            y_test_nb)
rate_nb = tpr_10fpr(tpr_nb, fpr_nb)
#%% 
logistic_bl = LogisticRegression()
logistic_bl.fit(X_RUS_bl.values, y_RUS_bl.values)

auc_bl = get_auc(logistic_bl, X_val_sc_bl, y_val_bl)

#%% evaluation on test data
plot2, auc_test_bl, tpr_bl, fpr_bl, threshold_bl = plot_roc(logistic_bl, X_test_sc_bl,
                                                            y_test_bl)
rate_bl = tpr_10fpr(tpr_bl, fpr_bl)

#%% Create final figure
plt.figure(0)
plt.title('ROC Logistic Regression - Separate Models')
plt.plot(fpr_bl, tpr_bl, 'b', label = 'AUC - black = %0.3f' % auc_test_bl)
plt.plot(fpr_nb, tpr_nb, 'g', label = 'AUC - non black = %0.3f' % auc_test_nb)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('/Users/zeliedresse/Dropbox/DABE/Thesis/LR_results/ROC_race_separate.png')
