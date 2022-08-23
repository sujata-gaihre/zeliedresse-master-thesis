#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:44:39 2022

@author: zeliedresse
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Import dataset
df = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/data_v2_spon.pkl")

#%% Categories for each variable type
QUANT_VAR = ["MAGER", "PRIORLIVE", "PRIORDEAD", "PRIORTERM", "PRECARE",
             "CIG_0", "CIG_1", "CIG_2", "BMI", "RF_CESARN", "M_Ht_In", "PWgt_R"]

CAT_VAR  = ["RESTATUS", "MRACE6", "MHISP_R",  "MEDUC", "PAY_REC", "OEGest_R10", "ILLB_R11"]

BOOL_VAR = ["WIC", "CIG_REC", "RF_PDIAB", "RF_GDIAB", "RF_PHYPE", "RF_GHYPE",
            "RF_EHYPE", "RF_PPTERM", "RF_INFTR", "RF_FEDRG", "RF_ARTEC", "RF_CESAR", 
            "IP_GON", "IP_SYPH", "IP_CHLAM", "IP_HEPB", "IP_HEPC", "FEMALE", "father_info",
            "US_NATIVE", "young_mother", "old_mother", "low_educ", "high_educ", "BMI_low",
            "BMI_overweight", "BMI_obese", "first_preg", "first_birth", "DMAR_U", "DMAR_Y",
            "MAR_P_U", "MAR_P_Y", "no_precare", "precare_1trim", "precare_2trim"]

#%% Create summary statistics for both preterm and non-preterm birth
# Quant var
table_quant_preterm = df[df["preterm"] == True][QUANT_VAR].describe()
table_quant_rest =  df[df["preterm"] == False][QUANT_VAR].describe()

# Categorical var
count_RESTATUS = df.groupby("preterm")["RESTATUS"].value_counts(normalize = True, sort = False)
count_MRACE6 = df.groupby("preterm")["MRACE6"].value_counts(normalize = True, sort = False)
count_MHISP = df.groupby("preterm")["MHISP_R"].value_counts(normalize = True, sort = False)
count_MEDUC = df.groupby("preterm")["MEDUC"].value_counts(normalize = True, sort = False)
count_PAY_REC = df.groupby("preterm")["PAY_REC"].value_counts(normalize = True, sort = False)
count_OEGest = df.groupby("preterm")["OEGest_R10"].value_counts(normalize = True, sort = False)
count_ILLB = df.groupby("preterm")["ILLB_R11"].value_counts(normalize = True, sort = False)

# bool var
table_bool_preterm = df[df["preterm"] == True][BOOL_VAR].mean()
table_bool_rest = df[df["preterm"] == False][BOOL_VAR].mean()

#%% number of preterm births
print(df["preterm"].sum())
print(df.shape[0] - df["preterm"].sum())
