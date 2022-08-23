#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:15:41 2022

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

#%% CV set
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

#%% Tune number of estimators
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, cv, DMatrix

xgb_bl = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 500, max_depth = 5, min_child_weight = 1, 
                    subsample = 0.8, objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, colsample_bytree = 0.8)

param = xgb_bl.get_xgb_params()
xgtrain_bl = DMatrix(X_cv_bl.values, label = y_cv_bl.values)
cv_result = cv(param, xgtrain_bl, num_boost_round = xgb_bl.get_params()["n_estimators"] , folds = ps_bl, metrics = 'auc', early_stopping_rounds = 20, verbose_eval = 1)

#%% Tune max depth + min child weight
from sklearn.model_selection import GridSearchCV
import time
start = time.time()
param_test1 = {'max_depth': [5, 7, 9, 11],
               'min_child_weight': [1, 3, 5]}

xgb_bl2 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 102, 
                    subsample = 0.8, objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, colsample_bytree = 0.8)
gsearch1 = GridSearchCV(xgb_bl2, param_grid = param_test1, scoring = 'roc_auc', n_jobs = -1, cv = ps_bl)
gsearch1.fit(X_cv_bl, y_cv_bl)

end = time.time()
print(end-start)

# max depth: 7, min weight: 5
#%%
param_test2 = {'max_depth': [6, 7, 8],
               'min_child_weight': [4, 5, 8, 10]}

gsearch2 = GridSearchCV(xgb_bl2, param_grid = param_test2, scoring = 'roc_auc', n_jobs = -1, cv = ps_bl)
gsearch2.fit(X_cv_bl, y_cv_bl)
# 6 en 8
#%%
param_test2b = {'max_depth': [5, 6, 7],
               'min_child_weight': [7, 8, 9]}

gsearch2b = GridSearchCV(xgb_bl2, param_grid = param_test2b, scoring = 'roc_auc', n_jobs = -1, cv = ps_bl)
gsearch2b.fit(X_cv_bl, y_cv_bl)

# max_depth = 6 , child_weight = 8
#%% gamma
param_test3 = {'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5]}

xgb_bl3 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 102, max_depth= 6, min_child_weight= 8,
                    subsample = 0.8, objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, colsample_bytree = 0.8)
gsearch3 = GridSearchCV(xgb_bl3, param_grid = param_test3, scoring = 'roc_auc', n_jobs = -1, cv = ps_bl)
gsearch3.fit(X_cv_bl, y_cv_bl)

# gamma = 0
#%% Apply early stopping 
xgb_bl4 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 500, max_depth= 6, min_child_weight= 9,
                    subsample = 0.8, objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, colsample_bytree = 0.8, gamma = 0)

param = xgb_bl4.get_xgb_params()
cv_result = cv(param, xgtrain_bl, num_boost_round = xgb_bl4.get_params()["n_estimators"] , folds = ps_bl, metrics = 'auc', early_stopping_rounds = 20, verbose_eval = 1)
#87
#%% Tune subsample and colsample
xgb_bl5 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 62, max_depth= 6, min_child_weight= 8,
                    objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, gamma = 0)

param_test4 = {'subsample': [0.2, 0.4, 0.6, 0.8, 1],
               'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1]}
gsearch4 = GridSearchCV(xgb_bl5, param_test4, scoring = 'roc_auc', n_jobs = -1, cv = ps_bl)
gsearch4.fit(X_cv_bl, y_cv_bl)

# subsample 0.6, colsample = 0.8

#%%
param_test5 = {'subsample': [0.5, 0.6, 0.7],
               'colsample_bytree': [0.7, 0.8, 0.9]}
gsearch5 = GridSearchCV(xgb_bl5, param_test5, scoring = 'roc_auc', n_jobs = -1, cv = ps_bl)
gsearch5.fit(X_cv_bl, y_cv_bl)
# subsample 0.5, colsample = 0.8


#%% l2 regularization
xgb_bl6 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 62, max_depth= 6, min_child_weight= 8,
                    subsample = 0.5, objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, colsample_bytree = 0.8, gamma = 0)

param_test6 = {'reg_lambda': [0, 0.01, 0.1, 1, 10, 100]}
gsearch6 = GridSearchCV(xgb_bl6, param_test6, scoring = 'roc_auc', n_jobs = -1, cv = ps_bl)
gsearch6.fit(X_cv_bl, y_cv_bl)

#%%
param_test7 = {'reg_lambda': [0.05, 0.1, 0.15, 0.2]}
gsearch7 = GridSearchCV(xgb_bl6, param_test7, scoring = 'roc_auc', n_jobs = -1, cv = ps_bl)
gsearch7.fit(X_cv_bl, y_cv_bl)

#0.1
#%% Apply early stopping + try lower learning rate
xgb_bl7 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.05, 
                    n_estimators = 500, max_depth= 6, min_child_weight= 8,
                    subsample = 0.5, objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, colsample_bytree = 0.8, gamma = 0, reg_lambda = 0.1)

param = xgb_bl7.get_xgb_params()
cv_result = cv(param, xgtrain_bl, num_boost_round = xgb_bl7.get_params()["n_estimators"] , folds = ps_bl, metrics = 'auc', early_stopping_rounds = 20, verbose_eval = 1)

#%% Final model black
from functions import plot_roc, tpr_10fpr
xgb_bl_final = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.05, 
                    n_estimators = 190, max_depth= 6, min_child_weight= 8,
                    subsample = 0.5, objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, colsample_bytree = 0.8, gamma = 0, reg_lambda = 0.1)
xgb_bl_final.fit(X_train_bl, y_train_bl)

plot_bl, auc_bl, tpr_bl, fpr_bl, threshold_bl = plot_roc(xgb_bl_final, X_test_bl, y_test_bl)
rate_bl = tpr_10fpr(tpr_bl, fpr_bl) 

#%% Non-black: tune number of estimators
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, cv, DMatrix

xgb_nb = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 500, max_depth = 5, min_child_weight = 1, 
                    subsample = 0.8, objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, colsample_bytree = 0.8)

param = xgb_nb.get_xgb_params()
xgtrain_nb = DMatrix(X_cv_nb.values, label = y_cv_nb.values)
cv_result = cv(param, xgtrain_nb, num_boost_round = xgb_nb.get_params()["n_estimators"] , folds = ps_nb, metrics = 'auc', early_stopping_rounds = 20, verbose_eval = 1)

#%% Tune depth and weight
xgb_nb2 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 194, 
                    subsample = 0.8, objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, colsample_bytree = 0.8)

param_test1 = {'max_depth': [5, 6, 7, 8, 9, 11],
               'min_child_weight': [1, 3, 4, 5, 6]}
gridsearch1 = GridSearchCV(xgb_nb2, param_test1, scoring = "roc_auc", n_jobs = -1 , cv = ps_nb)
gridsearch1.fit(X_cv_nb, y_cv_nb)

# max depth = 5, min_weight = 4

#%% 
param_test2 = {'max_depth': [3, 4, 5],
               'min_child_weight': [3, 4, 5]}

gridsearch2 = GridSearchCV(xgb_nb2, param_test2, scoring = "roc_auc", n_jobs = -1 , cv = ps_nb)
gridsearch2.fit(X_cv_nb, y_cv_nb)

# max depth = 5, min weight = 4

#%% Tune gamma
xgb_nb3 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 194, max_depth = 5, min_child_weight = 4, 
                    subsample = 0.8, objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, colsample_bytree = 0.8)

param_test3 = {'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5]}
gridsearch3 = GridSearchCV(xgb_nb3, param_test3, scoring = "roc_auc", n_jobs = -1 , cv = ps_nb)
gridsearch3.fit(X_cv_nb, y_cv_nb)

# gamma = 0.1

#%% Apply early stopping
xgb_nb4 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 500, max_depth = 5, min_child_weight = 4, 
                    subsample = 0.8, objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, colsample_bytree = 0.8, gamma = 0.1)

param = xgb_nb4.get_xgb_params()
xgtrain_nb = DMatrix(X_cv_nb.values, label = y_cv_nb.values)

cv_result = cv(param, xgtrain_nb, num_boost_round = xgb_nb4.get_params()["n_estimators"] , folds = ps_nb, metrics = 'auc', early_stopping_rounds = 20, verbose_eval = 1)

#%% Tune colsample by tree + subsample
xgb_nb5 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 210, max_depth = 5, min_child_weight = 4, 
                    objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, gamma = 0.1)

param_test4 = {'subsample': [0.5, 0.7, 0.9, 1],
               'colsample_bytree': [0.4, 0.6, 0.8]}

gridsearch4 = GridSearchCV(xgb_nb5, param_test4, scoring = "roc_auc", n_jobs = -1 , cv = ps_nb)
gridsearch4.fit(X_cv_nb, y_cv_nb)
#{'colsample_bytree': 0.4, 'subsample': 0.5}

#%%
param_test5 = {'subsample': [0.4, 0.5, 0.6],
               'colsample_bytree': [0.3, 0.4, 0.5]}
gridsearch5 = GridSearchCV(xgb_nb5, param_test5, scoring = "roc_auc", n_jobs = -1 , cv = ps_nb)
gridsearch5.fit(X_cv_nb, y_cv_nb)

#%%
param_test6 = {'subsample': [0.3, 0.4],
               'colsample_bytree': [0.4]}

gridsearch6 = GridSearchCV(xgb_nb5, param_test6, scoring = "roc_auc", n_jobs = -1 , cv = ps_nb)
gridsearch6.fit(X_cv_nb, y_cv_nb)

#0.4 , 0.4

#%% l2 regularization
xgb_nb6 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 210, max_depth = 5, min_child_weight = 4, 
                    objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, gamma = 0.1, colsample_bytree = 0.4, 
                    subsample  = 0.4)

param_test7 = {'reg_lambda': [0, 0.01, 0.1, 1, 10, 100]}
gridsearch7 = GridSearchCV(xgb_nb6, param_test7, scoring = "roc_auc", n_jobs = -1 , cv = ps_nb)
gridsearch7.fit(X_cv_nb, y_cv_nb)

#%%
param_test8 = {'reg_lambda': [0.05, 0.1, 0.15, 0.2]}
gridsearch8 = GridSearchCV(xgb_nb6, param_test8, scoring = "roc_auc", n_jobs = -1 , cv = ps_nb)
gridsearch8.fit(X_cv_nb, y_cv_nb)
#0.1

#%% Tune number of estimators again
xgb_nb7 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 500, max_depth = 5, min_child_weight = 4, 
                    objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, gamma = 0.1, colsample_bytree = 0.4, 
                    subsample  = 0.4, reg_lambda = 0.1)

param = xgb_nb7.get_xgb_params()
cv_result = cv(param, xgtrain_nb, num_boost_round = xgb_nb7.get_params()["n_estimators"] , folds = ps_nb, metrics = 'auc', early_stopping_rounds = 20, verbose_eval = 1)

#%% Final model non-black
xgb_nb_final = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.1, 
                    n_estimators = 217, max_depth = 5, min_child_weight = 4, 
                    objective = 'binary:logistic', n_jobs = -1,
                    seed = 8, gamma = 0.1, colsample_bytree = 0.4, 
                    subsample  = 0.4, reg_lambda = 0.1)
xgb_nb_final.fit(X_train_nb, y_train_nb)

plot_nb, auc_nb, tpr_nb, fpr_nb, threshold_nb = plot_roc(xgb_nb_final, X_test_nb, y_test_nb)
rate_nb = tpr_10fpr(tpr_nb, fpr_nb) 

#%% Plot ROC curves
plt.figure(0)
plt.title('ROC XGBoost - Separate Models')
plt.plot(fpr_bl, tpr_bl, 'b', label = 'AUC - black = %0.3f' % auc_bl)
plt.plot(fpr_nb, tpr_nb, 'g', label = 'AUC - non-black = %0.3f' % auc_nb)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/DT_results/ROC_XGB_race_separate.png")

#%% Check variable importance
feature_names = X_train_bl.columns
importances_bl = xgb_bl_final.feature_importances_

forest_importances_bl = pd.DataFrame({'importances': importances_bl} ,index = feature_names)
forest_importances_bl = forest_importances_bl.sort_values(by = "importances", ascending = False)

feature_names = X_train_nb.columns
importances_nb = xgb_nb_final.feature_importances_

forest_importances_nb = pd.DataFrame({'importances': importances_nb} ,index = feature_names)
forest_importances_nb = forest_importances_nb.sort_values(by = "importances", ascending = False)
#%% Variable importance black
plt.figure(0)
plt.barh(forest_importances_bl.index[:10], forest_importances_bl["importances"][:10])
plt.xticks(rotation = 0)
plt.ylabel("Mean decrease in impurity")
plt.xlabel("Features")
plt.grid(axis = "x", linestyle = '--', linewidth = 0.5)
plt.title("Variable Importance Random Forest - black")
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/DT_results/VarImp-XGB-Black.png",
            bbox_inches = "tight")
#%% Variable importance non-black
plt.figure(0)
plt.barh(forest_importances_nb.index[:10], forest_importances_nb["importances"][:10])
plt.xticks(rotation = 0)
plt.ylabel("Mean decrease in impurity")
plt.xlabel("Features")
plt.grid(axis = "x", linestyle = '--', linewidth = 0.5)
plt.title("Variable Importance Random Forest - non-black")
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/DT_results/VarImp-XGB-NB.png",
            bbox_inches = "tight")