#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:05:39 2022

@author: zeliedresse
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
#%% Import relevant columns from csv file
cols_to_use = ["MAGER", "MBSTATE_REC", "RESTATUS", "MRACE6", "MHISP_R", 
               "MEDUC", "PRIORLIVE", "PRIORDEAD","PRIORTERM",  "ILLB_R11",
               "PRECARE","WIC","CIG_0","CIG_1","CIG_2", "CIG_REC",
               "BMI","RF_PDIAB","RF_GDIAB","RF_PHYPE",
               "RF_GHYPE","RF_EHYPE","RF_INFTR","RF_FEDRG",
               "RF_ARTEC","RF_CESAR","RF_CESARN", "OEGest_R10", "SEX",
               'IP_HEPB', 'IP_HEPC', 'IP_SYPH', 'RF_PPTERM', 'IP_CHLAM','IP_GON', 
               "DMAR", "MAR_P", "PAY_REC", "FAGECOMB", "LD_INDL"]

VAR = ['M_Ht_In', 'PWgt_R', "ME_ROUT", "ME_TRIAL"]
filename = "/Users/zeliedresse/Documents/Thesis Data/birth_2020_nber_us.csv"

df = pd.read_csv(filename,sep = ",", usecols = cols_to_use, low_memory=False)
#%% Get other variables from txt file
""" CSV file on NBER website is missing certain variables,
retrieving them here from the txt file
"""

with open("/Users/zeliedresse/Documents/Thesis Data/Nat2020PublicUS.c20210506.r20210812.txt") as f:
    contents = f.readlines()

data = []
M_Ht_In = []
for i in range(len(contents)):
    M_Ht_In.append(contents[i][279:281])
data.append(M_Ht_In)

PWgt_R = []
for i in range(len(contents)):
    PWgt_R.append(contents[i][291:294])
data.append(PWgt_R)

ME_ROUT = []
for i in range(len(contents)):
    ME_ROUT.append(contents[i][401])
data.append(ME_ROUT)

ME_TRIAL = []
for i in range(len(contents)):
    ME_TRIAL.append(contents[i][402])
data.append(ME_TRIAL)

df_2 = pd.DataFrame(data)
df_2 = df_2.transpose()
df_2.columns = VAR

df_2["PWgt_R"] = pd.to_numeric(df_2["PWgt_R"])
df_2["M_Ht_In"] = pd.to_numeric(df_2["M_Ht_In"])

#%% Drop empty rows in first DF
nan_values = df[df.isna().all(axis=1)]      
nan_rows = nan_values.index.tolist()

df.drop(nan_rows, inplace= True)
df = df.reset_index()
#%% Put two dataframes together
df_final = pd.concat([df, df_2], axis=1)

#%% delete unneeded objects in memory
del contents, data, df, df_2, nan_rows, nan_values, i, PWgt_R, M_Ht_In, VAR,
del f, cols_to_use, filename, ME_ROUT, ME_TRIAL
#%% Downcast variables and convert to categorical
df = df_final.copy()
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

with pd.ExcelWriter('missing_value.xlsx') as writer:  
    missing.to_excel(writer, sheet_name='2020')
    
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
#%% drop columns
df = df.drop(columns = ["SEX","MBSTATE_REC", "FAGECOMB"])
df.reset_index(inplace = True)
#%% Outcome variable
df["preterm"] = np.where(df["OEGest_R10"] <=5, True, False)

#%% save file
#df.to_pickle("2020.pkl")
df.to_pickle("2020_v2.pkl")
