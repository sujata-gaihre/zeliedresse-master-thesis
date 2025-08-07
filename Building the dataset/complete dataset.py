#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:39:21 2022

@author: zeliedresse
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

#%% import different years
df_2020 = pd.read_pickle("Data/2020_v2.pkl")
df_2019 = pd.read_pickle("Data/2019_final_v2.pkl")
df_2018 = pd.read_pickle("Data/2018_v2.pkl")
df_2017 = pd.read_pickle("Data/2017_v2.pkl")
df_2016 = pd.read_pickle("Data/2016_v2.pkl")

#%% drop index columns
df_2020.drop(columns = ["level_0", "index"], inplace = True)
df_2019.drop(columns = ["level_0", "index"], inplace = True)
df_2018.drop(columns = ["level_0", "index"], inplace = True)
df_2017.drop(columns = ["level_0", "index"], inplace = True)
df_2016.drop(columns = ["level_0", "index"], inplace = True)

#%% merge dataframes
df = pd.concat([df_2020, df_2019, df_2018, df_2017, df_2016])
print(sys.getsizeof(df))

#%% transform columns to boolean
bool_col = ["WIC", "CIG_REC", "RF_PDIAB", "RF_GDIAB", "RF_PHYPE", "RF_GHYPE",
            "RF_EHYPE", "RF_PPTERM", "RF_INFTR", "RF_CESAR", "IP_GON", "IP_SYPH",
            "IP_CHLAM", "IP_HEPB", "IP_HEPC", "father_info", "RF_FEDRG", "RF_ARTEC"]

df[bool_col] = df[bool_col].replace({"Y": 1, "N": 0})
df[bool_col] = df[bool_col].astype(bool)

#%% recode RACE
df["MRACE6"] = df["MRACE6"].replace({10: 1, 20: 2, 30: 3, 40: 4, 
                                     41: 4, 51: 5, 61: 6})

#%% float to int
fcols = df.select_dtypes('float64').columns

def float_to_int( s ):
    if ( s.astype(np.int64) == s ).all():
        return pd.to_numeric( s, downcast='integer' )
    else:
        return s

df[fcols] = df.loc[:, fcols].apply( float_to_int )
print(sys.getsizeof(df)) 

#%% object to category
object_cols = df.select_dtypes(include=['object']).columns 

for i in object_cols:
    df[i] = df[i].astype('category')
    
#%% new birth_interval
df["birth_interval"] = df["ILLB_R11"] + 1
df["birth_interval"] = df["birth_interval"].replace({89:0})

#%% new variables
df["young_mother"] = df["MAGER"].apply(lambda x: True if x < 20 else False)
df["old_mother"] = df["MAGER"].apply(lambda x: True if x >= 35 else False)

df["low_educ"] = df["MEDUC"].apply(lambda x: True if x <= 3 else False)
df["high_educ"] = df["MEDUC"].apply(lambda x: True if x > 6 else False)

df["BMI_low"] = df["BMI"].apply(lambda x: True if x < 18.5 else False)
df["BMI_overweight"] = df["BMI"].apply(lambda x: True if (x >= 25) & (x < 35) else False)
df["BMI_obese"] = df["BMI"].apply(lambda x: True if x > 35 else False)

df["sum_preg"] = df["PRIORLIVE"] + df["PRIORDEAD"] + df["PRIORTERM"]
df["sum_birth"]= df["PRIORLIVE"] + df["PRIORDEAD"]
df["first_preg"] = df["sum_preg"].apply(lambda x: True if x == 0 else False)
df["first_birth"] = df["sum_birth"].apply(lambda x: True if x == 0 else False)
df.drop(columns = ["sum_preg", "sum_birth"], inplace = True)

#%% make dummies of DMAR and MAR_P
df = pd.get_dummies(df, columns = ["DMAR", "MAR_P"], drop_first = True)

#%% new prenatal care variable
df["no_precare"] = df["PRECARE"] == 0
df["precare_1trim"] = (df["PRECARE"] >= 1) & (df["PRECARE"] <= 3)
df["precare_2trim"] =  (df["PRECARE"] >= 4) & (df["PRECARE"] <= 6)
#df["precare_3trim"] = df["PRECARE"] >= 7

#%% full dataset
#df.to_pickle("data_v2.pkl")

#%% only spontaneous births
df["ME_ROUT"] = pd.to_numeric(df["ME_ROUT"])

df_vag_spon = df[(df["ME_ROUT"] != 4) & (df["LD_INDL"] == "N")]
df_caes = df[(df["ME_ROUT"] == 4) & (df["LD_INDL"] == "N") & (df["ME_TRIAL"] == "Y")]

df_spon = pd.concat([df_vag_spon, df_caes], ignore_index=True)
#%%
df_spon.to_pickle("Data/data_v2_spon.pkl")

