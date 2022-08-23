#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:41:24 2022

@author: zeliedresse
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

#%% Import scaled and undersampled data
X_train = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_train.pkl")
y_train = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_train.pkl")
X_val = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_val.pkl")
y_val = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_val.pkl")
X_test = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/X_test.pkl")
y_test = pd.read_pickle("/Users/zeliedresse/Documents/Thesis Data/dataframes/y_test.pkl")

#%% Initial model
model1 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model1_fit = model1.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Looking at learning rate
plt.plot(model1_fit.history['auc'])
plt.plot(model1_fit.history['val_auc'])
plt.title('model auc')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#tf.keras.callbacks.EarlyStopping(patience=20)
# val_auc 0.6946249604225159
#%% Comparing different learning rates
model1 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])
model1_fast_fit = model1.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model1 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
              loss='binary_crossentropy',
              metrics=['AUC'])
model1_slow_fit = model1.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%%
model1 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['AUC'])
model1_extrafast_fit = model1.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))
#%% Plotting models with different learning rates
fig, axs = plt.subplots(2,2, sharey = 'all', sharex= "all",figsize=(10,10))
axs[0,0].plot(model1_extrafast_fit.history['auc'])
axs[0,0].plot(model1_extrafast_fit.history['val_auc'])
axs[0,0].set_title("extra fast")
axs[0,1].plot(model1_fast_fit.history['auc'])
axs[0,1].plot(model1_fast_fit.history['val_auc'])
axs[0,1].set_title("fast")
axs[1,0].plot(model1_fit.history['auc'])
axs[1,0].plot(model1_fit.history['val_auc'])
axs[1,0].set_title("normal")
axs[1,1].plot(model1_slow_fit.history['auc'])
axs[1,1].plot(model1_slow_fit.history['val_auc'])
axs[1,1].set_title("slow")
plt.show()

#%%
plt.figure(0)
plt.plot(model1_fast_fit.history['auc'], '--r')
plt.plot(model1_fast_fit.history['val_auc'], 'r')
plt.plot(model1_fit.history['auc'], '--b')
plt.plot(model1_fit.history['val_auc'],'b')
plt.show()
#%% Looking at batch size
model1 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model1_256_fit = model1.fit(X_train, y_train, batch_size = 256, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model1 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model1_128_fit = model1.fit(X_train, y_train, batch_size = 128, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model1 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model1_1024_fit = model1.fit(X_train, y_train, batch_size = 1024, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Plotting models with different batch size
fig, axs = plt.subplots(2,2, sharey = 'all', sharex= "all",figsize=(10,10))
axs[0,0].plot(model1_fast_fit.history['auc'])
axs[0,0].plot(model1_fast_fit.history['val_auc'])
axs[0,0].set_title("512")
axs[0,1].plot(model1_1024_fit.history['auc'])
axs[0,1].plot(model1_1024_fit.history['val_auc'])
axs[0,1].set_title("1024")
axs[1,0].plot(model1_128_fit.history['auc'])
axs[1,0].plot(model1_128_fit.history['val_auc'])
axs[1,0].set_title("128")
axs[1,1].plot(model1_256_fit.history['auc'])
axs[1,1].plot(model1_256_fit.history['val_auc'])
axs[1,1].set_title("256")
plt.show()
# learning rate 0.001 and batch_size 512
#  0.6962065100669861

#%% Tweaking layers and number of neurons
model2 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model2_fit = model2.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))
#0.6963080763816833
#%% 
model3 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model3_fit = model3.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#0.6959342360496521

#%%
model4 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model4_fit = model4.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))
#0.6965670585632324

#%%
model5 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model5.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model5_fit = model5.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%%
model6 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model6.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model6_fit = model6.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Different weight initializers
from tensorflow.keras import initializers

model7 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])

model7.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model7_fit = model7.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model8 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomUniform(seed = 8)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomUniform(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomUniform(seed = 8))
])
model8.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model8_fit = model8.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model9= keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.GlorotNormal(seed = 8)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.GlorotNormal(seed = 8)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.GlorotNormal(seed = 8))
])

model9.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model9_fit = model9.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model10 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.GlorotUniform(seed = 8)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.GlorotUniform(seed = 8)),

    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.GlorotUniform(seed = 8))
])

model10.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model10_fit = model10.fit(X_train, y_train, batch_size = 512, epochs=50, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Plotting models with different kernel initializer
fig, axs = plt.subplots(2,3, sharey = 'all', sharex= "all",figsize=(10,10))
axs[0,2].plot(model6_fit.history['auc'])
axs[0,2].plot(model6_fit.history['val_auc'])
axs[0,2].set_title("default")
axs[0,0].plot(model7_fit.history['auc'])
axs[0,0].plot(model7_fit.history['val_auc'])
axs[0,0].set_title("random normal")
axs[1,0].plot(model8_fit.history['auc'])
axs[1,0].plot(model8_fit.history['val_auc'])
axs[1,0].set_title("random uniform")
axs[0,1].plot(model9_fit.history['auc'])
axs[0,1].plot(model9_fit.history['val_auc'])
axs[0,1].set_title("glorot normal")
axs[1,1].plot(model10_fit.history['auc'])
axs[1,1].plot(model10_fit.history['val_auc'])
axs[1,1].set_title("glorot uniform")
plt.show()

    # random normal
#%%
plt.figure()
plt.plot(model6_fit.history['val_auc'][:10], label = "default")
plt.plot(model7_fit.history['val_auc'][:10], label = 'random normal')
plt.plot(model8_fit.history['val_auc'][:10], label = 'random uniform')
plt.plot(model9_fit.history['val_auc'][:10], label = 'glorot normal')
plt.plot(model10_fit.history['val_auc'][:10], label = 'glorot uniform')
plt.legend()
plt.show()

#%% l2 regularization
from tensorflow.keras import regularizers
model7_0 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0))
])

model7_0.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model7_0_fit = model7_0.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))
#%%
model7_001 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.01)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])

model7_001.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model7_001_fit = model7_001.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))
#%%
model7_01 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.1)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.1)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])

model7_01.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model7_01_fit = model7_01.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model7_1 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(1)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(1)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])

model7_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model7_1_fit = model7_1.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model7_10 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(10)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(10)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])

model7_10.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model7_10_fit = model7_10.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))



#%%
model7_0001 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])

model7_0001.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model7_0001_fit = model7_0001.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model7_00001 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.0001)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.0001)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])

model7_00001.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model7_00001_fit = model7_00001.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Plotting models with different learning rates
plt.figure()
plt.plot(model7_0_fit.history['val_auc'], label = "0")
plt.plot(model7_00001_fit.history['val_auc'], label = '0.0001')
plt.plot(model7_0001_fit.history['val_auc'], label = '0.001')
plt.plot(model7_001_fit.history['val_auc'], label = '0.01')
plt.plot(model7_01_fit.history['val_auc'], label = '0.1')
plt.plot(model7_1_fit.history['val_auc'], label = '1')
plt.plot(model7_10_fit.history['val_auc'], label = '10')
plt.legend()
plt.show()

# 0.001
#%%
plt.figure()
plt.plot(model7_0001_fit.history['auc'])
plt.plot(model7_0001_fit.history['val_auc'])
plt.show()

#%% Other weight initializers
model7_henormal = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.HeNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.HeNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.HeNormal(seed = 8))
])

model7_henormal.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model7_henormal_fit = model7_henormal.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model7_heuniform = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.HeUniform(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.HeUniform(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.HeUniform(seed = 8))
])

model7_heuniform.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model7_heuniform_fit = model7_heuniform.fit(X_train, y_train, batch_size = 512, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))
#%% Plot models with new weight initializers
plt.figure()
plt.plot(model7_0001_fit.history['val_auc'], label = '0.001')
plt.plot(model7_heuniform_fit.history['val_auc'], label = 'he uniform')
plt.plot(model7_henormal_fit.history['val_auc'], label = 'he normal')
plt.legend()
plt.show()

#%% Different batch size
model7_256 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])

model7_256.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model7_256_fit = model7_256.fit(X_train, y_train, batch_size = 256, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

model7_128 = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])

model7_128.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

model7_128_fit = model7_128.fit(X_train, y_train, batch_size = 128, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

#%% Plotting again with different batch size
plt.figure()
plt.plot(model7_0001_fit.history['val_auc'], label = '512')
plt.plot(model7_256_fit.history['val_auc'], label = '256')
plt.plot(model7_128_fit.history['val_auc'], label = '128')
plt.legend()
plt.show()

# 256
#%% Final model
final_model = keras.Sequential([
    keras.layers.Input(shape=(51,)),
    keras.layers.Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(seed = 8),
                       kernel_regularizer = regularizers.l2(0.001)),
    keras.layers.Dense(1, activation='sigmoid',
                       kernel_initializer=initializers.RandomNormal(seed = 8))
])

final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC'])

final_fit = final_model.fit(X_train, y_train, batch_size = 128, epochs=100, 
                        validation_data = (X_val, y_val),
                        callbacks = tf.keras.callbacks.EarlyStopping(patience=20))

prob = final_model.predict(X_test)

#%% Evaluating final model
from sklearn.metrics import roc_auc_score, roc_curve
from functions import tpr_10fpr
auc_nn = roc_auc_score(y_test, prob)
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, prob)
rate_nn = tpr_10fpr(tpr_nn, fpr_nn)
#%% Plot ROC curve
plt.figure(5)
plt.title('ROC - NN')
plt.plot(fpr_nn, tpr_nn, 'b', label = 'AUC = %0.3f' % auc_nn)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/DT_results/ROC_NN.png")

#%% Evaluate performance among subgroups 
y_test.reset_index(inplace = True, drop = True)

#%%
index_bl = X_test[X_test["MRACE6"] == 0.3995775420860196].index
index_nb = X_test[X_test["MRACE6"] != 0.3995775420860196].index

fpr_bl, tpr_bl, threshold_bl = roc_curve(y_test[index_bl], prob[index_bl]) 

fpr_nb, tpr_nb, threshold_nb = roc_curve(y_test[index_nb], prob[index_nb]) 

auc_bl = roc_auc_score(y_test[index_bl], prob[index_bl])
auc_nb = roc_auc_score(y_test[index_nb], prob[index_nb])

rate_bl = tpr_10fpr(tpr_bl, fpr_bl)
rate_nb = tpr_10fpr(tpr_nb, fpr_nb)

#%%
plt.figure(5)
plt.title('ROC Curve NN - by race')
plt.plot(fpr_nn, tpr_nn, color = "orange", label = 'AUC - overall = %0.3f' % auc_nn)
plt.plot(fpr_bl, tpr_bl, 'b', label = 'AUC - black = %0.3f' % auc_bl)
plt.plot(fpr_nb, tpr_nb, 'g', label = 'AUC - non black = %0.3f' % auc_nb)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("/Users/zeliedresse/Dropbox/DABE/Thesis/DT_results/ROC_NN_race.png")