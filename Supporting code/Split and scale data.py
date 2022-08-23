#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:42:54 2022

@author: zeliedresse

Create train, validation and test set that is standardized.

"""
#%% Import packages
import numpy as np
import pandas as pd
import os

#%% Import dataset
df = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/data_v2_spon.pkl")

#%% create train, test and validation data set
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

X = df.drop(columns = ["OEGest_R10", "preterm", "ILLB_R11", "PWgt_R", "BMI", "ME_ROUT", "ME_TRIAL", "LD_INDL", "PRECARE"])
y = df["preterm"]

X_train, X_2, y_train, y_2 = train_test_split(X, y, test_size = 0.4, random_state = 8)
X_val, X_test, y_val, y_test = train_test_split(X_2, y_2, test_size = 0.5, random_state = 8)

del X_2, y_2

#%% Standardization
from sklearn.preprocessing import StandardScaler

scaler_train = preprocessing.StandardScaler().fit(X_train)
X_train_sc = pd.DataFrame(scaler_train.transform(X_train),
                          columns =scaler_train.get_feature_names_out())

scaler_val = preprocessing.StandardScaler().fit(X_val)
X_val_sc = pd.DataFrame(scaler_val.transform(X_val), 
                        columns = scaler_val.get_feature_names_out())

scaler_test = preprocessing.StandardScaler().fit(X_test)
X_test_sc = pd.DataFrame(scaler_test.transform(X_test),
                         columns = scaler_test.get_feature_names_out())

os.chdir("/Users/zeliedresse/Documents/Thesis Data/dataframes")

#%% Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state = 8)
X_sc_RUS, y_RUS = rus.fit_resample(X_train_sc, y_train)

#%% Save scaled and undersampled data
X_sc_RUS.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_train.pkl")
X_val_sc.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_val.pkl")
X_test_sc.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_test.pkl")

y_RUS.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_train.pkl")
y_val.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_val.pkl")
y_test.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_test.pkl")

#%% Separate data by race
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

scaler_train = StandardScaler().fit(X_train_nb)
X_train_nb_sc = pd.DataFrame(scaler_train.transform(X_train_nb),
                          columns =scaler_train.get_feature_names_out())

scaler_val = StandardScaler().fit(X_val_nb)
X_val_nb_sc = pd.DataFrame(scaler_val.transform(X_val_nb), 
                        columns = scaler_val.get_feature_names_out())

scaler_test = StandardScaler().fit(X_test_nb)
X_test_nb_sc = pd.DataFrame(scaler_test.transform(X_test_nb),
                         columns = scaler_test.get_feature_names_out())

scaler_train = StandardScaler().fit(X_train_bl)
X_train_bl_sc = pd.DataFrame(scaler_train.transform(X_train_bl),
                          columns =scaler_train.get_feature_names_out())

scaler_val = StandardScaler().fit(X_val_bl)
X_val_bl_sc = pd.DataFrame(scaler_val.transform(X_val_bl), 
                        columns = scaler_val.get_feature_names_out())

scaler_test = StandardScaler().fit(X_test_bl)
X_test_bl_sc = pd.DataFrame(scaler_test.transform(X_test_bl),
                         columns = scaler_test.get_feature_names_out())
#%% Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state = 8)

X_train_nb_sc_RUS, y_train_nb_RUS = rus.fit_resample(X_train_nb_sc, y_train_nb)
X_train_bl_sc_RUS, y_train_bl_RUS = rus.fit_resample(X_train_bl_sc, y_train_bl)


#%% Save scaled and undersampled datasets by race
X_train_nb_sc_RUS.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_train_nb.pkl")
X_val_nb_sc.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_val_nb.pkl")
X_test_nb_sc.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_test_nb.pkl")

y_train_nb_RUS.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_train_nb.pkl")
y_val_nb.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_val_nb.pkl")
y_test_nb.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_test_nb.pkl")

X_train_bl_sc_RUS.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_train_bl.pkl")
X_val_bl_sc.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_val_bl.pkl")
X_test_bl_sc.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_test_bl.pkl")

y_train_bl_RUS.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_train_bl.pkl")
y_val_bl.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_val_bl.pkl")
y_test_bl.to_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_test_bl.pkl")


