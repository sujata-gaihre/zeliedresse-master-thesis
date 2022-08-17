#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:12:45 2022

@author: zeliedresse
"""

#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

#%% Import relevant columns from csv file
cols_to_use = ["mager", "mbstate_rec", "restatus", "mrace6", "mhisp_r", 
               "meduc", "priorlive", "priordead","priorterm",  "illb_r11",
               "precare","wic","cig_0","cig_1","cig_2", "cig_rec", 
               "m_ht_in","bmi","rf_pdiab","rf_gdiab","rf_phype",
               "rf_ghype","rf_ehype","rf_ppterm","rf_inftr","rf_fedrg","rf_artec",
               "rf_cesar","rf_cesarn","ip_gon","ip_syph","ip_chlam","ip_hepatb",
               "ip_hepatc", "oegest_r10", "sex", "pwgt_r",
               "dmar", "mar_p", "pay_rec", "fagecomb", "ld_indl", "me_rout", "me_trial"]


filename = "/Users/zeliedresse/Documents/Thesis Data/natl2017.csv"

df = pd.read_csv(filename,sep = ",", usecols = cols_to_use, low_memory=False)
#%% Rename columns so they are consistent across years
df.columns = df.columns.str.upper()
df = df.rename({'M_HT_IN': 'M_Ht_In', 'PWGT_R': 'PWgt_R', 'IP_HEPATB': 'IP_HEPB',
                'IP_HEPATC': 'IP_HEPC', 'OEGEST_R10': 'OEGest_R10'}, axis='columns')

#%% Drop empty rows
nan_values = df[df.isna().all(axis=1)]      
nan_rows = nan_values.index.tolist()

df.drop(nan_rows, inplace= True)
df = df.reset_index()

#%% Downcast variables and convert to categorical
fcols = df.select_dtypes('float').columns
icols = df.select_dtypes('integer').columns

df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

object_cols = df.select_dtypes(include=['object']).columns 

for i in object_cols:
    df[i] = df[i].astype('category')

#%% Recode missing values
NA_U = ["WIC", "CIG_REC", "RF_PDIAB","RF_GDIAB",
        "RF_PHYPE","RF_GHYPE","RF_EHYPE","RF_PPTERM",
        "RF_INFTR","RF_FEDRG","RF_ARTEC","RF_CESAR",
        "IP_GON","IP_SYPH","IP_CHLAM","IP_HEPB","IP_HEPC"]
df[NA_U] = df[NA_U].replace({"U": np.NaN})

df["MAR_P"] = df["MAR_P"].replace({"X": "Y"})
df['MAR_P'] = df['MAR_P'].fillna("U")
df['DMAR'] = df['DMAR'].fillna("U")
df["DMAR"] = df["DMAR"].replace({1.0: "Y"})
df["DMAR"] = df["DMAR"].replace({2.0: "N"})

df["RF_FEDRG"] = df["RF_FEDRG"].replace({"X": "N"})
df["RF_ARTEC"] = df["RF_ARTEC"].replace({"X": "N"})

NA_3 = ["MBSTATE_REC"]
df[NA_3] = df[NA_3].replace({3.0:np.NAN})

NA_9 = ["MEDUC", "MHISP_R", "PAY_REC"]
df[NA_9] = df[NA_9].replace({9.0: np.NaN})

NA_99 = ["PRIORLIVE", "PRIORDEAD", "PRIORTERM", "ILLB_R11", "PRECARE", "CIG_0", "CIG_1",
         "CIG_2", "RF_CESARN", "OEGest_R10", "M_Ht_In"]
df[NA_99] = df[NA_99].replace({99.0: np.NaN})

df["BMI"] = df["BMI"].replace({99.9:np.NaN})
df["PWgt_R"] = df["PWgt_R"].replace({999:np.NaN})

#%% get table with NA
missing = df.isna().sum()

with pd.ExcelWriter('/Users/zeliedresse/missing_value.xlsx', engine='openpyxl',mode='a') as writer:   
    missing.to_excel(writer, sheet_name='2017')
    
#%% Delete all missing
df = df.dropna()

#%% 2 categories yes/no to bool
bool_col = ["WIC", "CIG_REC", "RF_PDIAB", "RF_GDIAB", "RF_PHYPE", "RF_GHYPE",
            "RF_EHYPE", "RF_PPTERM", "RF_INFTR", "RF_CESAR", "IP_GON", "IP_SYPH",
            "IP_CHLAM", "IP_HEPB", "IP_HEPC"]

df["FEMALE"] = df["SEX"].replace({'F':1, 'M':0}).astype(bool)

mask = df["FAGECOMB"] == 99.0
df["father_info"] = "Y"
df.loc[mask, "father_info"] = "N"

df[bool_col] = df[bool_col].replace({"Y": 1, "N": 0})
df[bool_col] = df[bool_col].astype(bool)
#%% Float to integer
fcols = df.select_dtypes('float32').columns

def float_to_int( s ):
    if ( s.astype(np.int64) == s ).all():
        return pd.to_numeric( s, downcast='integer' )
    else:
        return s

df[fcols] = df.loc[:, fcols].apply( float_to_int )

df["US_NATIVE"] = df["MBSTATE_REC"].replace({1:True, 2:False}).astype(bool)
#%% Drop columns
df = df.drop(columns = ["SEX","MBSTATE_REC", "FAGECOMB"])
df.reset_index(inplace = True)

#%% Outcome variable
df["preterm"] = np.where(df["OEGest_R10"] <=5, True, False)

#%% save file
#df.to_pickle("2017.pkl")
df.to_pickle("2017_v2.pkl")