#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:47:30 2022

@author: zeliedresse

Quick check how models perform when applying them only to first pregnancies. 
Due to time constraints only the final models from the complete datasets are applied to this dataset, without further tuning.
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Import dataset
df = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/data_v2_spon.pkl")

#%%
df = df[df["first_preg"]==True]

#%% create train, test and validation data set
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

X = df.drop(columns = ["OEGest_R10", "preterm", "ILLB_R11", "PWgt_R", "BMI", "ME_ROUT", "ME_TRIAL", "LD_INDL", "PRECARE", "birth_interval", "PRIORDEAD", "PRIORLIVE", "PRIORTERM",
                       "RF_PPTERM",  "RF_CESAR", "RF_CESARN"])
y = df["preterm"]

X_train, X_2, y_train, y_2 = train_test_split(X, y, test_size = 0.4, random_state = 8)
X_val, X_test, y_val, y_test = train_test_split(X_2, y_2, test_size = 0.5, random_state = 8)

del X_2, y_2

#%% Standardization and normalization
# Standardization
scaler_train = preprocessing.StandardScaler().fit(X_train)
X_train_sc = pd.DataFrame(scaler_train.transform(X_train),
                          columns =scaler_train.get_feature_names_out())

scaler_val = preprocessing.StandardScaler().fit(X_val)
X_val_sc = pd.DataFrame(scaler_val.transform(X_val), 
                        columns = scaler_val.get_feature_names_out())

scaler_test = preprocessing.StandardScaler().fit(X_test)
X_test_sc = pd.DataFrame(scaler_test.transform(X_test),
                         columns = scaler_test.get_feature_names_out())

#%% Random Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state = 8)

X_sc_RUS, y_RUS = rus.fit_resample(X_train_sc, y_train)

#%% Logistic regression final model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score, roc_auc_score
from functions import plot_roc
logistic5 = LogisticRegression()
logistic5.fit(X_sc_RUS, y_RUS)

plot_LR, auc_LR, tpr_LR, fpr_LR, threshold = plot_roc(logistic5, X_test_sc, y_test)
#%% Variable importance logistic regression
importance_LR = pd.DataFrame(logistic5.coef_[0])
importance_LR.set_index(X_sc_RUS.columns, inplace = True)

#%% RF final
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 21, random_state = 8, n_jobs = -1,
                                n_estimators = 100, criterion = "gini",
                                max_features = 9, verbose = 2)
rf.fit(X_sc_RUS, y_RUS)
plot_rf, auc_rf, tpr_rf, fpr_rf, threshold = plot_roc(rf, X_test_sc, y_test)

#%% XGB final
from xgboost import XGBClassifier
xgb_final =  XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.2,
                     n_estimators = 233, objective = 'binary:logistic',
                     seed = 8, max_depth = 6, min_child_weight  = 4, 
                     gamma = 0, n_jobs = -1, colsample_bytree = 0.5, 
                     subsample = 0.9, reg_lambda = 80)

xgb_final.fit(X_sc_RUS, y_RUS)

plot_xgb, auc_xgb, tpr_xgb, fpr_xgb, threshold = plot_roc(xgb_final, X_test_sc, y_test)

#%% NN final
import tensorflow as tf
from tensorflow import keras
from keras import initializers, regularizers

NN = keras.Sequential([
    keras.layers.Input(shape=(44,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])

NN.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

NN.fit(X_sc_RUS, y_RUS, batch_size = 128, epochs=100, 
                        validation_data = (X_val_sc, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))
#%%
prob_NN = NN.predict(X_test_sc)

auc_NN = roc_auc_score(y_test, prob_NN)
fpr_NN, tpr_NN, thresholds_NN = roc_curve(y_test, prob_NN)

#%% Plotting ROC curves together
plt.figure(5)
plt.title('ROC Curve - Random Forest')
plt.plot(fpr_LR, tpr_LR, color = "orange", label = 'AUC LR = %0.3f' % auc_LR)
plt.plot(fpr_rf, tpr_rf, 'b', label = 'AUC RF = %0.3f' % auc_rf)
plt.plot(fpr_xgb, tpr_xgb, 'g', label = 'AUC XGB = %0.3f' % auc_xgb)
plt.plot(fpr_NN, tpr_NN, color = "black", label = 'AUC NN = %0.3f' % auc_NN)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

#%% Calculating TPR at 10% FPR
from functions import tpr_10fpr
tpr10_LR = tpr_10fpr(tpr_LR, fpr_LR)
tpr10_RF = tpr_10fpr(tpr_rf, fpr_rf)
tpr10_xgb = tpr_10fpr(tpr_xgb, fpr_xgb)
tpr10_NN = tpr_10fpr(tpr_NN, fpr_NN)