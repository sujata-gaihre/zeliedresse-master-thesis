#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:09:39 2022

@author: zeliedresse
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
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
model1 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model1_fit = model1.fit(X_train_bl, y_train_bl, batch_size = 512, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))
#%% Different batch size
model2 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model2_fit = model2.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Different learning rate
model3 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model3_fit = model3.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Different batch size 
model4 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model4_fit = model4.fit(X_train_bl, y_train_bl, batch_size = 64, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Different learning rate
model5 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model5.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['AUC'])

model5_fit = model5.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))
#%%
plt.figure(0)
plt.plot(model2_fit.history["auc"], 'b')
plt.plot(model5_fit.history["auc"], 'r')
plt.plot(model2_fit.history["val_auc"], '--b')
plt.plot(model5_fit.history["val_auc"], '--r')
plt.show()

#%%
plt.figure(0)
plt.plot(model2_fit.history["loss"], 'b')
plt.plot(model5_fit.history["loss"], 'r')
plt.plot(model2_fit.history["val_loss"], '--b')
plt.plot(model5_fit.history["val_loss"], '--r')
plt.show()

# batch size 128, learning rate 0.001

#%% Changing number of layers and neurons
model6 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model6.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model6_fit = model6.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%%
model6 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model6.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model6_fit = model6.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%%
model7 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model7.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model7_fit = model7.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))
#%%
model8 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model8.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model8_fit = model8.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))
#%%
plt.figure(0)
plt.plot(model2_fit.history["auc"], 'b')
plt.plot(model8_fit.history["auc"], 'r')
plt.plot(model2_fit.history["val_auc"], '--b')
plt.plot(model8_fit.history["val_auc"], '--r')
plt.show()

#%%
plt.figure(0)
plt.plot(model2_fit.history["loss"], 'b')
plt.plot(model7_fit.history["loss"], 'r')
plt.plot(model2_fit.history["val_loss"], '--b')
plt.plot(model7_fit.history["val_loss"], '--r')
plt.show()

#%% Trying different weight initializers
from keras import initializers
model_RandomNormal = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])
model_RandomNormal.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_RandomNormal_fit = model_RandomNormal.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model_RandomUniform = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomUniform(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomUniform(seed = 8))
])
model_RandomUniform.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_RandomUniform_fit = model_RandomUniform.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model_GlorotNormal = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.GlorotNormal(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.GlorotNormal(seed = 8))
])
model_GlorotNormal.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_GlorotNormal_fit = model_GlorotNormal.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model_GlorotUniform = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.GlorotUniform(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.GlorotUniform(seed = 8))
])
model_GlorotUniform.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_GlorotUniform_fit = model_GlorotUniform.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model_HeNormal = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.HeNormal(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.HeNormal(seed = 8))
])
model_HeNormal.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_HeNormal_fit = model_HeNormal.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model_HeUniform = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.HeUniform(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.HeUniform(seed = 8))
])
model_HeUniform.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_HeUniform_fit = model_HeUniform.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Plot models with different initializers together
plt.figure(0)
plt.plot(model_RandomNormal_fit.history["val_auc"], label = 'random normal')
plt.plot(model_RandomUniform_fit.history["val_auc"], label = 'random uniform')
plt.plot(model_GlorotNormal_fit.history["val_auc"], label = 'glorot normal')
plt.plot(model_GlorotUniform_fit.history["val_auc"], label = 'glorot uniform')
plt.plot(model_HeNormal_fit.history["val_auc"], label = 'he normal')
plt.plot(model_HeUniform_fit.history["val_auc"], label = 'he uniform')
plt.legend()
plt.show()
# random uniform

#%% l2 regularization
from keras import regularizers
model_0 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomUniform(seed = 8),
                       kernel_regularizer = regularizers.l2(0)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomUniform(seed = 8))
])
model_0.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_0_fit = model_0.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model_0001 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomUniform(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomUniform(seed = 8))
])
model_0001.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_0001_fit = model_0001.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))


model_001 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomUniform(seed = 8),
                       kernel_regularizer = regularizers.l2(0.01)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomUniform(seed = 8))
])
model_001.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_001_fit = model_001.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))


model_01 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomUniform(seed = 8),
                       kernel_regularizer = regularizers.l2(0.1)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomUniform(seed = 8))
])
model_01.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_01_fit = model_01.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))


model_1 = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomUniform(seed = 8),
                       kernel_regularizer = regularizers.l2(1)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomUniform(seed = 8))
])
model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_1_fit = model_1.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Plotting models with different levels of regularization
plt.figure(0)
plt.plot(model_0_fit.history["val_auc"], label = '0')
plt.plot(model_0001_fit.history["val_auc"], label = '0.001')
plt.plot(model_001_fit.history["val_auc"], label = '0.01')
plt.plot(model_01_fit.history["val_auc"], label = '0.1 ')
plt.plot(model_1_fit.history["val_auc"], label = '1')
plt.legend()
plt.show()

#%% Final model black
from sklearn.metrics import roc_auc_score, roc_curve
from functions import tpr_10fpr
final_model_bl = keras.Sequential([
    keras.layers.Input(shape=(50,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomUniform(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomUniform(seed = 8))
    ])

final_model_bl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

final_fit_bl = final_model_bl.fit(X_train_bl, y_train_bl, batch_size = 128, epochs=100, 
                        validation_data = (X_val_bl, y_val_bl),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

prob_bl = final_model_bl.predict(X_test_bl)
auc_bl = roc_auc_score(y_test_bl, prob_bl)
fpr_bl, tpr_bl, threshold_bl = roc_curve(y_test_bl, prob_bl)
rate_bl = tpr_10fpr(tpr_bl, fpr_bl)

#%% Non-black group - initial model
model1 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model1_fit = model1.fit(X_train_nb, y_train_nb, batch_size = 512, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Different batch size
model2 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model2_fit = model2.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Different batch size
model3 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model3_fit = model3.fit(X_train_nb, y_train_nb, batch_size = 256, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))
#%%
plt.figure(0)
plt.plot(model2_fit.history["auc"], 'b', label = 'old')
plt.plot(model4_fit.history["auc"], 'r', label = 'new')
plt.plot(model2_fit.history["val_auc"], '--b')
plt.plot(model4_fit.history["val_auc"], '--r')
plt.legend()
plt.show()

# 128, 0.001

#%% Changing layers/neurons
model3 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid')
])
model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model3_fit = model3.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%%
model4 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model4_fit = model4.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Trying different weight initializers
from keras import initializers

model_RandomNormal = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])
model_RandomNormal.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_RandomNormal_fit = model_RandomNormal.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model_RandomUniform = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomUniform(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomUniform(seed = 8))
])
model_RandomUniform.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_RandomUniform_fit = model_RandomUniform.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model_GlorotNormal = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.GlorotNormal(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.GlorotNormal(seed = 8))
])
model_GlorotNormal.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_GlorotNormal_fit = model_GlorotNormal.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model_GlorotUniform = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.GlorotUniform(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.GlorotUniform(seed = 8))
])
model_GlorotUniform.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_GlorotUniform_fit = model_GlorotUniform.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model_HeNormal = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.HeNormal(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.HeNormal(seed = 8))
])
model_HeNormal.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_HeNormal_fit = model_HeNormal.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model_HeUniform = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.HeUniform(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.HeUniform(seed = 8))
])
model_HeUniform.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_HeUniform_fit = model_HeUniform.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Plotting models with different initializers
plt.figure(0)
plt.plot(model_RandomNormal_fit.history["val_auc"], label = 'random normal')
plt.plot(model_RandomUniform_fit.history["val_auc"], label = 'random uniform')
plt.plot(model_GlorotNormal_fit.history["val_auc"], label = 'glorot normal')
plt.plot(model_GlorotUniform_fit.history["val_auc"], label = 'glorot uniform')
plt.plot(model_HeNormal_fit.history["val_auc"], label = 'he normal')
plt.plot(model_HeUniform_fit.history["val_auc"], label = 'he uniform')
plt.legend()
plt.show()

# random normal

#%% l2 regularization
from keras import regularizers
model_0 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])
model_0.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_0_fit = model_0.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model_0001 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])
model_0001.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_0001_fit = model_0001.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))


model_001 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.01)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])
model_001.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_001_fit = model_001.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))


model_01 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.1)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])
model_01.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_01_fit = model_01.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))


model_1 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(1)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])
model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model_1_fit = model_1.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Plotting models with different levels of regularization
plt.figure(0)
plt.plot(model_0_fit.history["val_auc"], label = '0')
plt.plot(model_0001_fit.history["val_auc"], label = '0.001')
plt.plot(model_001_fit.history["val_auc"], label = '0.01')
plt.plot(model_01_fit.history["val_auc"], label = '0.1 ')
plt.plot(model_1_fit.history["val_auc"], label = '1')
plt.legend()
plt.show()

#%% Final model non-black
from sklearn.metrics import roc_auc_score, roc_curve
from functions import tpr_10fpr
final_model_nb = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
    ])

final_model_nb.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

final_fit_nb = final_model_nb.fit(X_train_nb, y_train_nb, batch_size = 128, epochs=100, 
                        validation_data = (X_val_nb, y_val_nb),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))
#%% Evaluating model
prob_nb = final_model_nb.predict(X_test_nb)
auc_nb = roc_auc_score(y_test_nb, prob_nb)
fpr_nb, tpr_nb, threshold_nb = roc_curve(y_test_nb, prob_nb)
rate_nb = tpr_10fpr(tpr_nb, fpr_nb)
