#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:13:03 2022

@author: zeliedresse
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Import standardized and undersampled data
X_train_nb = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_train_nb.pkl")
y_train_nb = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_train_nb.pkl")
X_val_nb = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_val_nb.pkl")
y_val_nb = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_val_nb.pkl")
X_test_nb = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_test_nb.pkl")
y_test_nb  = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_test_nb.pkl")

X_train_bl = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_train_bl.pkl")
y_train_bl = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_train_bl.pkl")
X_val_bl = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_val_bl.pkl")
y_val_bl = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_val_bl.pkl")
X_test_bl = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_test_bl.pkl")
y_test_bl  = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_test_bl.pkl")

#%% Initial model
from sklearn.ensemble import RandomForestClassifier
from functions import plot_roc

rf_bl = RandomForestClassifier(max_depth = 7, random_state = 8, n_jobs = -1,
                                n_estimators = 100, criterion = "gini", verbose = 2)
rf_bl.fit(X_train_bl, y_train_bl)

plot1, auc_bl, tpr_bl, frp_bl, threshold_bl = plot_roc(rf_bl, X_val_bl, y_val_bl)

#%% Using help functions to evaluate effect of changing mtry and depth
from functions import depth_plot, mtry_plot
plot1 = depth_plot(range(1,32,2), X_train_bl, y_train_bl, X_val_bl, y_val_bl)
plt.show()

#%%
plot2 = mtry_plot(range(1,12,2), X_train_bl, y_train_bl, X_val_bl, y_val_bl, max_depth = 17)
plt.show()
#%% Creating alternative dataframe for gridsearch with predefined splits
from sklearn.model_selection import PredefinedSplit
X_cv_bl = pd.concat([X_train_bl, X_val_bl], axis = 0)
X_cv_nb = pd.concat([X_train_nb, X_val_nb], axis = 0)

y_val_bl.reset_index(inplace = True, drop = True)
y_val_nb.reset_index(inplace = True, drop = True)
y_cv_bl = pd.concat([y_train_bl, y_val_bl], axis = 0)
y_cv_nb = pd.concat([y_train_nb, y_val_nb], axis = 0)

test_fold_bl = np.zeros(X_cv_bl.shape[0])
test_fold_bl[:X_train_bl.shape[0] + 1] = -1

ps_bl = PredefinedSplit(test_fold_bl)
ps_bl.get_n_splits()

test_fold_nb = np.zeros(X_cv_nb.shape[0])
test_fold_nb[:X_train_nb.shape[0] + 1] = -1

ps_nb = PredefinedSplit(test_fold_nb)
ps_nb.get_n_splits()

#%% Gridsearch to tune max_features and max_depth
from sklearn.model_selection import GridSearchCV
rf_bl2  = RandomForestClassifier(random_state = 8, n_jobs = -1, n_estimators = 100,
                                 criterion = "entropy")

param_test1 = {'max_features': [5, 6, 7, 8, 9],
               'max_depth': [16, 17, 18, 19]}
gsearch1 = GridSearchCV(rf_bl2, param_test1, scoring = 'roc_auc', cv = ps_bl)
gsearch1.fit(X_cv_bl, y_cv_bl)

# 19 en8

#%%
param_test2 = {'max_features': [7, 8, 9],
               'max_depth': [19, 20, 22, 25]}
gsearch2 = GridSearchCV(rf_bl2, param_test2, scoring = 'roc_auc', cv = ps_bl)
gsearch2.fit(X_cv_bl, y_cv_bl)

# 19.8
#%% Final model
rf_bl_final = RandomForestClassifier(max_depth = 19, random_state = 8, n_jobs = -1,
                                n_estimators = 100, criterion = "entropy",
                                max_features= 7, verbose = 2)
rf_bl_final.fit(X_train_bl, y_train_bl)
#%% Evaluating model on test data
from functions import tpr_10fpr

plot3, auc_bl_final, tpr_bl, fpr_bl, threshold_bl = plot_roc(rf_bl_final, X_test_bl, y_test_bl)
rate_bl = tpr_10fpr(tpr_bl, fpr_bl)

#%% Same process for non-black group
rf_nb = RandomForestClassifier(max_depth = 7, random_state = 8, n_jobs = -1,
                                n_estimators = 100, criterion = "entropy", verbose = 2)
rf_nb.fit(X_train_nb, y_train_nb)

plot4, auc_nb, tpr_nb, fpr_nb, threshold_nb = plot_roc(rf_nb, X_val_nb, 
                                                                 y_val_nb)

#%%
plot1 = depth_plot(range(1,32,2), X_train_nb, y_train_nb, X_val_nb, y_val_nb)
plt.show()

#%%
plot2 = mtry_plot(range(1,12,2), X_train_nb, y_train_nb, X_val_nb, y_val_nb, max_depth = 20)
plt.show()

#%%
rf_nb2  = RandomForestClassifier(random_state = 8, n_jobs = -1, n_estimators = 100,
                                 criterion = "entropy")

param_test1 = {'max_features': [6, 7, 8, 9, 10],
               'max_depth': [18, 19, 20, 21, 22]}
gsearch1 = GridSearchCV(rf_nb2, param_test1, scoring = 'roc_auc', cv = ps_nb)
gsearch1.fit(X_cv_nb, y_cv_nb)
## 22 en 8

#%%
param_test2 = {'max_features': [7, 8, 9],
               'max_depth': [22, 24, 26]}
gsearch2 = GridSearchCV(rf_nb2, param_test2, scoring = 'roc_auc', cv = ps_nb)
gsearch2.fit(X_cv_nb, y_cv_nb)

# 22 en 9

#%%
param_test3 = {'max_features':  [9, 10],
               'max_depth': [22]}
gsearch3 = GridSearchCV(rf_nb2, param_test3, scoring = 'roc_auc', cv = ps_nb)
gsearch3.fit(X_cv_nb, y_cv_nb)
## 22 en 9
#%%
rf_nb_final = RandomForestClassifier(max_depth = 22, random_state = 9, n_jobs = -1,
                                n_estimators = 100, criterion = "entropy",
                                max_features= 9, verbose = 2)
rf_nb_final.fit(X_train_nb, y_train_nb)

#%%
plot3, auc_nb_final, tpr_nb, fpr_nb, threshold_nb = plot_roc(rf_nb_final, X_test_nb, y_test_nb)

rate_nb = tpr_10fpr(tpr_nb, fpr_nb)
#%% Plotting both models together
plt.figure(0)
plt.title('ROC Random Forest - Separate Models')
plt.plot(fpr_bl, tpr_bl, 'b', label = 'AUC - black = %0.3f' % auc_bl_final)
plt.plot(fpr_nb, tpr_nb, 'g', label = 'AUC - non-black = %0.3f' % auc_nb_final)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/DT_results/ROC_RF_race_separate.png")

#%% Exploring variable importance
feature_names = X_train_bl.columns
importances_bl = rf_bl_final.feature_importances_
std_bl = np.std([tree.feature_importances_ for tree in rf_bl_final.estimators_], axis=0)

forest_importances_bl = pd.DataFrame({'importances': importances_bl, 'std': std_bl} ,index = feature_names)
forest_importances_bl = forest_importances_bl.sort_values(by = "importances", ascending = False)

feature_names = X_train_nb.columns
importances_nb = rf_nb_final.feature_importances_
std_nb = np.std([tree.feature_importances_ for tree in rf_nb_final.estimators_], axis=0)

forest_importances_nb = pd.DataFrame({'importances': importances_nb, 'std': std_nb} ,index = feature_names)
forest_importances_nb = forest_importances_nb.sort_values(by = "importances", ascending = False)

#%%
labels_bl = ["Birth Interval", "Previous Preterm", "Age", "Height", "Education",
             "Prior births (alive)", "Resident Status", "Prior other preg.", 
             "No prenatal care", "Gest. Hypertension"]
labels_nb = ["Birth Interval", "Previous Preterm", "Age", "Height", "Education",
             "Prior births (alive)",  "Gest. Hypertension", "Prior other preg.",
             "Payment Method", "Hispanic Origin"]
#%% Plotting variable importance for black group
plt.figure(0)
plt.barh(labels_bl, forest_importances_bl["importances"][:10])
plt.xticks(rotation = 0)
plt.ylabel("Mean decrease in impurity")
plt.xlabel("Features")
plt.grid(axis = "x", linestyle = '--', linewidth = 0.5)
plt.title("Variable Importance Random Forest - black")
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/DT_results/VarImp-Black.png",
            bbox_inches = "tight")
#%% Plotting variable importance for non-black group
plt.figure(0)
plt.barh(labels_nb, forest_importances_nb["importances"][:10])
plt.xticks(rotation = 0)
plt.ylabel("Mean decrease in impurity")
plt.xlabel("Features")
plt.grid(axis = "x", linestyle = '--', linewidth = 0.5)
plt.title("Variable Importance Random Forest - non-black")
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/DT_results/VarImp-NB.png",
            bbox_inches = "tight")