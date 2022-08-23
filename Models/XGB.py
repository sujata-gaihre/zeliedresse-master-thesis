#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:27:59 2022

@author: zeliedresse
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Import scaled and undersampled data
X_train = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_train.pkl")
y_train = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_train.pkl")
X_val = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_val.pkl")
y_val = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_val.pkl")
X_test = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_test.pkl")
y_test = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_test.pkl")

#%% Creating alternative dataframe for gridsearch with predefined splits
from sklearn.model_selection import PredefinedSplit
X_cv = pd.concat([X_train, X_val], axis = 0)
y_val.reset_index(inplace = True, drop = True)
y_cv = pd.concat([y_train, y_val], axis = 0)

test_fold = np.zeros(X_cv.shape[0])
test_fold[:X_train.shape[0]] = -1

ps = PredefinedSplit(test_fold)
ps.get_n_splits()

#%% Initial model + tuning number of estimators
from xgboost import XGBClassifier, cv, DMatrix
xgb = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.2, 
                    n_estimators = 500, max_depth = 5, min_child_weight = 1, 
                    subsample = 0.8, objective = 'binary:logistic', n_jobs = 7,
                    seed = 8)

param = xgb.get_xgb_params()
xgtrain = DMatrix(X_cv.values, label = y_cv.values)
cv_result = cv(param, xgtrain, num_boost_round = xgb.get_params()["n_estimators"] , folds = ps, metrics = 'auc', early_stopping_rounds = 20, verbose_eval = 1)
#%% 
from functions import plot_roc
xgb.set_params(n_estimators = cv_result.shape[0])
xgb.fit(X_train, y_train)
plot1, auc1, tpr1, fpr1, threshold1 = plot_roc(xgb, X_val, y_val)
#%% Tuning max depth and min child weight
from sklearn.model_selection import GridSearchCV
import time
start = time.time()
param_test1 = {'max_depth': [5, 7, 9, 11],
               'min_child_weight': [1, 3, 5]}

xgb2 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.2,
                     n_estimators = 117, subsample = 0.8, objective = 'binary:logistic',
                     colsample_bytree = 0.8, seed = 8)
gsearch1 = GridSearchCV(xgb2, param_grid = param_test1, scoring = 'roc_auc', n_jobs = -1, cv = ps)
gsearch1.fit(X_cv, y_cv)

end = time.time()
print(end-start)
#40 min
# 7 en 5
#%%
start = time.time()
param_test2 = {'max_depth': [6, 7, 8],
               'min_child_weight': [4, 5, 6]}

xgb2 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.2,
                     n_estimators = 117, subsample = 0.8, objective = 'binary:logistic',
                     colsample_bytree = 0.8, seed = 8)
gsearch2 = GridSearchCV(xgb2, param_grid = param_test2, scoring = 'roc_auc', n_jobs = -1, cv = ps)
gsearch2.fit(X_cv, y_cv)
end = time.time()
print(end-start)
# 30 min voor 9x
# max_depth = 6, min_child_weight = 4
#%%
start = time.time()
param_test2b = {'max_depth': [5, 6],
               'min_child_weight': [3, 4]}


gsearch2b = GridSearchCV(xgb2, param_grid = param_test2b, scoring = 'roc_auc', n_jobs = -1, cv = ps)
gsearch2b.fit(X_cv, y_cv)
end = time.time()
print(end-start)
# 30 min voor 9x
# max_depth = 6, min_child_weight = 4
#%% Tuning Gamma
import time
start = time.time()
param_test3 = {'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5]}
xgb3 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.2,
                     n_estimators = 117, subsample = 0.8, objective = 'binary:logistic',
                     seed = 8, max_depth = 6, min_child_weight  = 4, colsample_bytree = 0.8)
gsearch3 = GridSearchCV(xgb3, param_grid = param_test3, scoring = 'roc_auc', 
                        n_jobs = -1, cv = ps)
gsearch3.fit(X_cv, y_cv)
end = time.time()
print(end-start)

#24 min
# gamma 0

#%%
start = time.time()

xgb4 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.2,
                     n_estimators = 300, subsample = 0.8, objective = 'binary:logistic',
                     seed = 8, max_depth = 6, min_child_weight  = 4, colsample_bytree = 0.8,
                     gamma = 0, n_jobs = -1)

param = xgb4.get_xgb_params()
cv_result = cv(param, xgtrain, num_boost_round = xgb4.get_params()["n_estimators"] , folds = ps, metrics = 'auc', early_stopping_rounds = 20, verbose_eval = 1)
end = time.time()
print(end-start)
xgb4.set_params(n_estimators = cv_result.shape[0])
#100
#%% Tuning subsample and colsamply by tree
start = time.time()
xgb4 =  XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.2,
                     n_estimators = 167, objective = 'binary:logistic',
                     seed = 8, max_depth = 6, min_child_weight  = 4, 
                     gamma = 0, n_jobs = -1)
param_test4 = {'subsample': [0.6, 0.7, 0.8, 0.9 ],
               'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]}
gsearch4 = GridSearchCV(xgb4, param_grid = param_test4, scoring = 'roc_auc', 
                        n_jobs = -1, cv = ps)
gsearch4.fit(X_cv, y_cv)
end = time.time()
print(end-start)

# 36 min
# colsample_bytree 0.6, subsample: 0.9
#%%
start = time.time()

param_test5 = {'subsample': [0.8, 0.9, 1],
              'colsample_bytree': [0.4, 0.5, 0.6]}
gsearch5 = GridSearchCV(xgb4, param_grid = param_test5, scoring = 'roc_auc', 
                        n_jobs = -1, cv = ps)
gsearch5.fit(X_cv, y_cv)
end = time.time()
print(end-start)

#subsample 0.9, colsample 0.5


#%% l2 regularization
start = time.time()
xgb6 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.2,
                     n_estimators = 167, objective = 'binary:logistic',
                     seed = 8, max_depth = 6, min_child_weight  = 4, 
                     gamma = 0, n_jobs = -1, colsample_bytree = 0.5, subsample = 0.9)

param_test6 = {'reg_lambda': [0, 0.01, 0.1, 1, 10, 100]}
gsearch6 = GridSearchCV(xgb6, param_grid = param_test6, scoring = 'roc_auc', 
                        n_jobs = -1, cv = ps)
gsearch6.fit(X_cv, y_cv)
end = time.time()
print(end-start)
# 10

#%%
start = time.time()
param_test6 = {'reg_lambda': [80, 90, 100, 110]}
gsearch6 = GridSearchCV(xgb6, param_grid = param_test6, scoring = 'roc_auc', 
                        n_jobs = -1, cv = ps)
gsearch6.fit(X_cv, y_cv)
end = time.time()
print(end-start)

#%%
start = time.time()
param_test6 = {'reg_lambda': [50, 70, 80]}
gsearch6 = GridSearchCV(xgb6, param_grid = param_test6, scoring = 'roc_auc', 
                        n_jobs = -1, cv = ps)
gsearch6.fit(X_cv, y_cv)
end = time.time()
print(end-start)

# 80
#%% Tune number of estimators again
xgb7 = XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.2,
                     n_estimators = 500, objective = 'binary:logistic',
                     seed = 8, max_depth = 6, min_child_weight  = 4, 
                     gamma = 0, n_jobs = -1, colsample_bytree = 0.5, 
                     subsample = 0.9, reg_lambda = 80)

param = xgb7.get_xgb_params()
cv_result = cv(param, xgtrain, num_boost_round = xgb7.get_params()["n_estimators"] , folds = ps, metrics = 'auc', early_stopping_rounds = 20, verbose_eval = 1)

#%% Final model
xgb_final =  XGBClassifier(random_state = 8, eval_metric = 'auc', learning_rate = 0.2,
                     n_estimators = 233, objective = 'binary:logistic',
                     seed = 8, max_depth = 6, min_child_weight  = 4, 
                     gamma = 0, n_jobs = -1, colsample_bytree = 0.5, 
                     subsample = 0.9, reg_lambda = 80)

xgb_final.fit(X_train, y_train)
#%%
from functions import tpr_10fpr, plot_roc
plot_xgb, auc_xgb, tpr_xgb, fpr_xgb, threshold_xgb = plot_roc(xgb_final, X_test, y_test)
rate = tpr_10fpr(tpr_xgb, fpr_xgb)
#%% Plot final ROC
plt.figure(5)
plt.title('ROC - XGBoost')
plt.plot(fpr_xgb, tpr_xgb, 'b', label = 'AUC = %0.3f' % auc_xgb)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/DT_results/ROC_XGB.png")

#%% variable importance
importances = xgb_final.feature_importances_
feature_names = X_train.columns
xgb_importances = pd.DataFrame({'importances': importances} ,index = feature_names)
xgb_importances = xgb_importances.sort_values(by = "importances", ascending = False)

labels = ["Height", "Age", "Birth Interval", "Prior births (alive)", "Education",
          "Prior other preg.", "Race", "Payment Method", "Hispanic Origin", "Cig. before preg."]

plt.figure(0)
plt.barh(labels, xgb_importances["importances"][:10])
plt.xticks(rotation = 0)
plt.xlabel("Mean decrease in impurity")
plt.ylabel("Features")
plt.grid(axis = "x", linestyle = '--', linewidth = 0.5)
plt.title("Variable Importance - XGBoost")
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/DT_results/VarImp-XGB",
            bbox_inches = "tight")

#%% evaluate performance on each subgroup
y_test.reset_index(inplace = True, drop = True)

index_bl = X_test[X_test["MRACE6"] == 0.3995775420860196].index
index_nb = X_test[X_test["MRACE6"] != 0.3995775420860196].index

#%%
from sklearn.metrics import roc_auc_score, roc_curve
probabilities = xgb_final.predict_proba(X_test)[:,1]

fpr_bl, tpr_bl, threshold = roc_curve(y_test[index_bl], probabilities[index_bl]) 

fpr_nb, tpr_nb, threshold = roc_curve(y_test[index_nb], probabilities[index_nb]) 

auc_bl = roc_auc_score(y_test[index_bl], probabilities[index_bl])
auc_nb = roc_auc_score(y_test[index_nb], probabilities[index_nb])

rate_bl = tpr_10fpr(tpr_bl, fpr_bl)
rate_nb = tpr_10fpr(tpr_nb, fpr_nb)
#%%
plt.figure(5)
plt.title('ROC Curve XGB - by race')
plt.plot(fpr_xgb, tpr_xgb, '--r', label = 'AUC - overall = %0.3f' % auc_xgb)
plt.plot(fpr_bl, tpr_bl, 'b', label = 'AUC - black = %0.3f' % auc_bl)
plt.plot(fpr_nb, tpr_nb, 'g', label = 'AUC - non black = %0.3f' % auc_nb)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/DT_results/ROC_XGB_race.png")