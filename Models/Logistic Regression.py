#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:33:22 2022

@author: zeliedresse

First exploration of logistic regression. The following things are explored with:
    - standardization versus normalization
    - dealing with class imbalance by using sampling methods and class weights 
    - feature selection
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Normalization
from sklearn.preprocessing import MinMaxScaler
minmax_train = MinMaxScaler().fit(X_train)
X_train_mm = pd.DataFrame(minmax_train.transform(X_train),
                          columns =minmax_train.get_feature_names_out())

minmax_val = MinMaxScaler().fit(X_val)
X_val_mm = pd.DataFrame(minmax_val.transform(X_val),
                          columns =minmax_val.get_feature_names_out())

minmax_test = MinMaxScaler().fit(X_test)
X_test_mm = pd.DataFrame(minmax_test.transform(X_test),
                          columns =minmax_test.get_feature_names_out())

#%% Undersampling
# Random undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state = 8)

X_sc_RUS, y_RUS = rus.fit_resample(X_train_sc, y_train)
X_mm_RUS, y_RUS = rus.fit_resample(X_train_mm, y_train)

#%% Other sampling methods: Near Miss and TomekLinks
"""
Not used because computation times are too long
# Near miss
from imblearn.under_sampling import NearMiss
nm1 = NearMiss(version = 1)
X_resampled_nm1, y_resampled_nm1 = nm1.fit_resample(X_train_small, y_train_small)

from imblearn.under_sampling import TomekLinks 
tl = TomekLinks()
X_res_tl, y_res_tl = tl.fit_resample(X_train_small, y_train_small)
"""
#%% Feature selection
### Variance Threshold
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold = 0.05)
sel.fit(X_mm_RUS)

X_sel = X_mm_RUS.loc[:, sel.get_support()]

### Select KBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

# chi-square
selKBest1_mm = SelectKBest(chi2, k = 20).fit(X_mm_RUS, y_RUS)
X_selKBest1_mm = X_mm_RUS.loc[:, selKBest1_mm.get_support()]

# ANOVA f-value
selKBest2_sc = SelectKBest(f_classif, k = 20).fit(X_sc_RUS, y_RUS)
X_selKBest2_sc = X_sc_RUS.loc[:, selKBest2_sc.get_support()]

selKBest2_mm = SelectKBest(f_classif, k = 20).fit(X_mm_RUS, y_RUS)
X_selKBest2_mm = X_mm_RUS.loc[:, selKBest2_mm.get_support()]

# selects same columns
#%% Trying models
from sklearn.linear_model import LogisticRegression
from functions import get_auc

# standardization
logistic1 = LogisticRegression()
logistic1.fit(X_train_sc, y_train)
pred1 = logistic1.predict(X_val_sc)

print(round(get_auc(logistic1, X_val_sc, y_val),4))

# normalization
logistic2 = LogisticRegression()
logistic2.fit(X_train_mm, y_train)

print(round(get_auc(logistic2, X_val_mm, y_val),4))

# stand + weights
logistic3 = LogisticRegression(class_weight = "balanced")
logistic3.fit(X_train_sc, y_train)

print(round(get_auc(logistic3, X_val_sc, y_val),4))

# norm + weights
logistic4 = LogisticRegression(class_weight = "balanced")
logistic4.fit(X_train_mm, y_train)

print(round(get_auc(logistic4, X_val_mm, y_val),4))

# stand + RUS
logistic5 = LogisticRegression()
logistic5.fit(X_sc_RUS, y_RUS)

print(round(get_auc(logistic5, X_val_sc, y_val),4))

# norm + RUS
logistic6 = LogisticRegression()
logistic6.fit(X_mm_RUS, y_RUS)

print(round(get_auc(logistic6, X_val_mm, y_val),4))

# norm + variance threshold
logistic7 = LogisticRegression()
logistic7.fit(X_sel.values, y_RUS.values)

print(round(get_auc(logistic7,sel.transform(X_val_mm), y_val),4))

# norm + chi2
logistic8 = LogisticRegression()
logistic8.fit(X_selKBest1_mm.values, y_RUS.values)

print(round(get_auc(logistic8, selKBest1_mm.transform(X_val_mm), y_val),4))

# stand + anova
logistic9 = LogisticRegression()
logistic9.fit(X_selKBest2_sc.values, y_RUS.values)

print(round(get_auc(logistic9, selKBest2_sc.transform(X_val_sc), y_val),4))

# norm + anova
logistic10 = LogisticRegression()
logistic10.fit(X_selKBest2_mm.values, y_RUS.values)

print(round(get_auc(logistic10, selKBest2_mm.transform(X_val_mm), y_val),4))

#%% selected model on test data
# decided on model 5: standardization + random undersampling
from functions import plot_roc, tpr_10fpr
plot1, auc5_test, tpr5, fpr5, threshold5 = plot_roc(logistic5, X_test_sc, y_test)

rate5 = tpr_10fpr(tpr5, fpr5)
#%% save ROC curve
plt.figure(0)
plt.title('ROC Curve - Logistic Regression')
plt.plot(fpr5, tpr5, 'b', label = 'AUC Model 1 = %0.3f' % auc5_test)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/LR_results/ROC_Model1.png")

#%% Number of features: F-ANOVA
# Exploring how feature selection affects results
auc_list = []
for i in range(1,52):
    KBest = SelectKBest(f_classif, k = i).fit(X_sc_RUS, y_RUS)
    X_KBest = X_sc_RUS.loc[:, KBest.get_support()]
    
    logistic = LogisticRegression()
    logistic.fit(X_KBest.values, y_RUS.values)
    auc_list.append(get_auc(logistic, KBest.transform(X_val_sc), y_val))

plt.figure(1)
plt.plot(range(1,52), auc_list)
plt.xlabel("Number of features")
plt.ylabel("Validation AUC")
plt.title("Feature selection with ANOVA F-value")
plt.savefig("LR_features_spon.png")

# clear jump is at birth interval

#%%  Number of features: variance thresholding
auc_list2 = []
threshold_list = np.linspace(0, 0.2, 21)
no_var = []
for i in threshold_list:
    var_sel = VarianceThreshold(threshold = i).fit(X_mm_RUS)
    X_var_sel = X_mm_RUS.loc[:, var_sel.get_support()]
    
    logistic = LogisticRegression()
    logistic.fit(X_var_sel.values, y_RUS.values)

    auc_list2.append(get_auc(logistic, var_sel.transform(X_val_mm), y_val))
    no_var.append(X_var_sel.shape[1])
    print(i)

plt.figure(2)
plt.plot(no_var, auc_list2)
plt.xlabel("Number of features")
plt.ylabel("Test AUC")
plt.title("Feature selection with Variance Thresholding")
plt.savefig("LR_variance_features_spon.png")

plt.figure(3)
plt.plot(threshold_list, auc_list2)
plt.xlabel("Threshold")
plt.ylabel("Test AUC")
plt.title("Feature selection with Variance Thresholding")
plt.savefig("LR_variance_threshold_spon.png")

#%% Model with 30 selected variables
anova30 = SelectKBest(f_classif, k = 30).fit(X_sc_RUS, y_RUS)
X_anova30 = X_sc_RUS.loc[:, anova30.get_support()]

logistic11 = LogisticRegression()
logistic11.fit(X_anova30.values, y_RUS.values)

roc_auc11 = get_auc(logistic11, anova30.transform(X_val_sc), y_val)
print(round(roc_auc11,4))

#%% 30 variables + birth interval
X_anova_31 = pd.concat([X_anova30, X_sc_RUS["birth_interval"]], axis = 1)

logistic12 = LogisticRegression()
logistic12.fit(X_anova_31.values, y_RUS.values)

print(round(get_auc(logistic12, X_val_sc[X_anova_31.columns], y_val),4))

#%% 30 var + birth interval on test data
plot2, auc12_test, tpr12, fpr12, threshold12 = plot_roc(logistic12,
                                                        X_test_sc[X_anova_31.columns],
                                                        y_test)

rate12 = tpr_10fpr(tpr12, fpr12)
#%% plot ROC curve with both models
plt.figure(1)
plt.title('ROC Curve - Logistic Regression')
plt.plot(fpr5, tpr5, 'b', label = 'AUC Model 1 = %0.3f' % auc5_test)
plt.plot(fpr12, tpr12, 'g', label = 'AUC Model 2 = %0.3f' % auc12_test)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/LR_results/ROC_LR.png")

#%% Variable importances of 30 var + birth interval
importance = pd.DataFrame(logistic12.coef_[0])
importance.set_index(X_anova_31.columns, inplace = True)

importance_abs = pd.DataFrame(abs(logistic12.coef_[0]))
importance_abs.set_index(X_anova_31.columns, inplace = True)
importance_abs = importance_abs.sort_values(by = 0, ascending = False)

index_largest = importance_abs.index[:10]

#%% plot 10 largest coefficients
labels = ["Previous Preterm", "Birth Interval", "Gest. Hypertension", "No Prenatal Care",
          "Gest. Diabetes", "Infertility Treat.", "Married", "Pre-preg. Hypertension",
          "Height", "Education"]
plt.figure(6)
plt.barh(labels, importance.loc[index_largest,0])
plt.xticks(rotation = 0)
plt.xlabel("Coefficients")
plt.ylabel("Features")
plt.grid(axis = "x", linestyle = '--', linewidth = 0.5)
plt.title("Logistic Regression - Model 2")
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/LR_results/coefficients_31.png", 
            bbox_inches = "tight")
# most important: RF_PPTERM, birth interval, RF_GHYPE, no_precare, RF_GDIAB

#%% Performance black versus non-black
X_test_sc.reset_index(inplace = True)
X_test_sc.drop(columns = ["index", "level_0"], inplace  = True)
y_test.reset_index(inplace = True, drop = True)

# find corresponding indices
index_bl = X_test_sc[X_test_sc["MRACE6"] == 0.3995775420860196].index
index_nb = X_test_sc[X_test_sc["MRACE6"] != 0.3995775420860196].index

#%%
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
probabilities = logistic5.predict_proba(X_test_sc)[:,1]

fpr_bl, tpr_bl, threshold = roc_curve(y_test[index_bl], probabilities[index_bl]) 
fpr_nb, tpr_nb, threshold = roc_curve(y_test[index_nb], probabilities[index_nb]) 

auc_bl = roc_auc_score(y_test[index_bl], probabilities[index_bl])
auc_nb = roc_auc_score(y_test[index_nb], probabilities[index_nb])

rate_bl = tpr_10fpr(tpr_bl, fpr_bl)
rate_nb = tpr_10fpr(tpr_nb, fpr_nb)

#%% Plot different ROC curves together
plt.figure(5)
plt.title('ROC Curve - Logistic Regression')
plt.plot(fpr5, tpr5,  color = "orange", label = 'AUC Model 1 = %0.3f' % auc5_test)
plt.plot(fpr_bl, tpr_bl, 'b', label = 'AUC Model 1 - black = %0.3f' % auc_bl)
plt.plot(fpr_nb, tpr_nb, 'g', label = 'AUC Model 1 - non black = %0.3f' % auc_nb)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('/Users/zeliedresse/Dropbox/DABE/Thesis/LR_results/ROC_race.png')

