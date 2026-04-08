#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *

import math
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy.matlib

import time
import os

# Create weight directories
os.makedirs('./initial_weights', exist_ok=True)
os.makedirs('./best_weights',    exist_ok=True)

# Reproducibility
from numpy.random import seed
seed(0)
tf.random.set_seed(0)

# Load & normalise data
# Generate synthetic wind/flood loss data if CSV not present (swap for real CSV)
try:
    df = pd.read_csv("wind_flood_loss_pairs2.csv")
    data_x = df["log_wind"].values.astype(np.float32).reshape(-1, 1)
    data_y = df["log_flood"].values.astype(np.float32).reshape(-1, 1)
    data   = np.concatenate((data_x, data_y), axis=1)
except FileNotFoundError:
    np.random.seed(42)
    n = 1000
    wind  = np.random.exponential(scale=1.0, size=n).astype(np.float32)
    flood = 0.6 * wind + 0.4 * np.random.exponential(scale=1.0, size=n).astype(np.float32)
    data  = np.column_stack([wind, flood])
    print("CSV not found — using synthetic data for demonstration.")

for i in range(data.shape[-1]):
    data[:, i] = (data[:, i] - np.min(data[:, i])) /                  (np.max(data[:, i]) - np.min(data[:, i]))

X_train, X_test = train_test_split(data, test_size=0.33, random_state=42)

# Key constants 
number_of_dimension       = 2
number_of_training_samples = X_train.shape[0]

# data_domain: [d x 2]
data_domain = np.array([[X_train[:, i].min(), X_train[:, i].max()]
                         for i in range(number_of_dimension)], dtype=np.float32)

# hyper-parameters 
num_hidden_layers_for_marginal_distribution  = 5
hidden_layer_width_for_marginal_distribution = 5
num_hidden_layers_for_copula                 = 5
hidden_layer_width_for_copula                = 10

# MODEL DEFINITIONS

class _marginal(keras.Model):
    def __init__(self):
        super(_marginal, self).__init__()
        self.kernel_regularizer = tf.keras.regularizers.L2(l2=0.001)
        self.dense_layer_list = [
            keras.layers.Dense(hidden_layer_width_for_marginal_distribution,
                               activation='tanh',
                               kernel_regularizer=self.kernel_regularizer)
            for _ in range(num_hidden_layers_for_marginal_distribution)
        ]
        self.final_layer  = keras.layers.Dense(1, activation='sigmoid',
                                               kernel_regularizer=self.kernel_regularizer)
        self.gradient_layers = Lambda(lambda x: K.gradients(x[0], x[1]))
        self.relu = keras.layers.ReLU()

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layer_list:
            x = layer(x)
        cdf = self.final_layer(x)
        pdf = self.gradient_layers([cdf, inputs])
        neg = self.relu(-pdf[0])
        pdf = 1e-9 + self.relu(pdf[0])
        return [cdf, pdf, neg]


class _copula(keras.Model):
    def __init__(self):
        super(_copula, self).__init__()
        self.kernel_regularizer = tf.keras.regularizers.L2(l2=0.001)
        self.get_components = Lambda(
            lambda x: [x[:, i:i+1] for i in range(number_of_dimension)])
        self.cat_components = Lambda(lambda x: K.concatenate(x, axis=-1))
        self.dense_layer_list = [
            keras.layers.Dense(hidden_layer_width_for_copula,
                               activation='tanh',
                               kernel_regularizer=self.kernel_regularizer)
            for _ in range(num_hidden_layers_for_copula)
        ]
        self.final_layer     = keras.layers.Dense(1, activation='sigmoid',
                                                   kernel_regularizer=self.kernel_regularizer)
        self.gradient_layers = Lambda(lambda x: K.gradients(x[0], x[1]))
        self.relu = keras.layers.ReLU()

    def call(self, inputs):
        components = self.get_components(inputs)
        x = self.cat_components(components)
        for layer in self.dense_layer_list:
            x = layer(x)
        cdf = self.final_layer(x)
        pdf = cdf
        for i in range(number_of_dimension):
            pdf = self.gradient_layers([pdf, components[i]])
        neg = self.relu(-pdf[0])
        pdf = 1e-9 + self.relu(pdf[0])
        return [cdf, pdf, neg]



marginal_model_list = [_marginal() for _ in range(number_of_dimension)]
copula_model        = _copula()

marginal_prediction_input_list  = [keras.layers.Input(shape=(1,))
                                    for _ in range(number_of_dimension)]
marginal_prediction_output_list = [marginal_model_list[i](marginal_prediction_input_list[i])
                                    for i in range(number_of_dimension)]
copula_prediction_input  = keras.layers.Input(shape=(number_of_dimension,))
copula_prediction_output = copula_model(copula_prediction_input)

marginal_prediction_model_list = [
    keras.Model(inputs=marginal_prediction_input_list[i],
                outputs=marginal_prediction_output_list[i])
    for i in range(number_of_dimension)
]
copula_prediction_model = keras.Model(inputs=copula_prediction_input,
                                       outputs=copula_prediction_output)


# MARGINAL TRAINING DATA

number_marginal_points = 1000

marginal_boundary_loss_data_list = [
    np.expand_dims(data_domain[i], axis=[0, -1])                       
    for i in range(number_of_dimension)
]
marginal_neg_sum_loss_data_list = [
    np.expand_dims(
        np.linspace(data_domain[i, 0], data_domain[i, 1], number_marginal_points),
        axis=[0, -1])                                                    
    for i in range(number_of_dimension)
]
marginal_log_loss_data_list = [
    np.expand_dims(X_train[:, i], axis=[0, -1])                         
    for i in range(number_of_dimension)
]

print('marginal_boundary_loss_data_list: d x', marginal_boundary_loss_data_list[0].shape)
print('marginal_neg_sum_loss_data_list:  d x', marginal_neg_sum_loss_data_list[0].shape)
print('marginal_log_loss_data_list:      d x', marginal_log_loss_data_list[0].shape)

marginal_training_input_data = (marginal_boundary_loss_data_list +
                                 marginal_neg_sum_loss_data_list  +
                                 marginal_log_loss_data_list)

marginal_boundary_loss_label_list = [np.expand_dims([0.0, 1.0], axis=0) for _ in range(number_of_dimension)]
marginal_neg_loss_label_list      = [np.expand_dims([0.0],       axis=0) for _ in range(number_of_dimension)]
marginal_sum_loss_label_list      = [np.expand_dims([1.0],       axis=0) for _ in range(number_of_dimension)]
marginal_log_loss_label_list      = [np.expand_dims([5.0],       axis=0) for _ in range(number_of_dimension)]

marginal_training_labels = (marginal_boundary_loss_label_list +
                             marginal_neg_loss_label_list      +
                             marginal_sum_loss_label_list      +
                             marginal_log_loss_label_list)

# Build marginal training model 
marginal_boundary_loss_input_list = [keras.layers.Input(shape=(2, 1))
                                      for _ in range(number_of_dimension)]
marginal_neg_sum_loss_input_list  = [keras.layers.Input(shape=(number_marginal_points, 1))
                                      for _ in range(number_of_dimension)]
marginal_log_loss_input_list      = [keras.layers.Input(shape=(number_of_training_samples, 1))
                                      for _ in range(number_of_dimension)]

marginal_loss_input_list = (marginal_boundary_loss_input_list +
                             marginal_neg_sum_loss_input_list  +
                             marginal_log_loss_input_list)

marginal_boundary_loss_output_list = [
    Lambda(lambda x: tf.expand_dims(x, axis=0), name="MB{}".format(i))(
        marginal_model_list[i](marginal_boundary_loss_input_list[i][0, :, :])[0][:, 0])
    for i in range(number_of_dimension)
]
marginal_neg_loss_output_list = [
    Lambda(lambda x: tf.math.reduce_sum(x, keepdims=True) / np.float32(number_marginal_points),
           name="MN{}".format(i))(
        marginal_model_list[i](marginal_neg_sum_loss_input_list[i][0, :, :])[2])
    for i in range(number_of_dimension)
]
marginal_sum_loss_output_list = [
    Lambda(lambda x: tf.math.reduce_sum(x, keepdims=True) *
           np.float32(data_domain[i, 1] - data_domain[i, 0]) / np.float32(number_marginal_points - 1),
           name="MS{}".format(i))(
        marginal_model_list[i](marginal_neg_sum_loss_input_list[i][0, :, :])[1])
    for i in range(number_of_dimension)
]
marginal_log_loss_output_list = [
    Lambda(lambda x: tf.math.reduce_sum(tf.math.log(x), keepdims=True) /
           np.float32(marginal_log_loss_input_list[0].shape[1]),
           name="ML{}".format(i))(
        marginal_model_list[i](marginal_log_loss_input_list[i][0, :, :])[1])
    for i in range(number_of_dimension)
]

marginal_loss_output_list = (marginal_boundary_loss_output_list +
                              marginal_neg_loss_output_list       +
                              marginal_sum_loss_output_list       +
                              marginal_log_loss_output_list)

marginal_model_train = keras.Model(inputs=marginal_loss_input_list,
                                    outputs=marginal_loss_output_list)

# Save initial weights (train 0 epochs to initialise variables first) 
marginal_model_train.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.001),
                              loss="mae")
marginal_model_train.fit(x=marginal_training_input_data,
                          y=marginal_training_labels,
                          epochs=1, verbose=0)
marginal_model_train.save_weights('./initial_weights/marginal_model_initial_weights.h5')

temp_history = marginal_model_train.fit(x=marginal_training_input_data,
                                         y=marginal_training_labels,
                                         epochs=1, verbose=0)
marginal_loss_keys = list(temp_history.history.keys())
print(marginal_loss_keys)

# Callback 
class Marginal_Model_Training_Callback(tf.keras.callbacks.Callback):
    def __init__(self, record_interval=10, show_interval=100, verbose=1):
        super().__init__()
        self.loss_keys      = marginal_loss_keys
        self.previous_loss  = 9999999
        self.record_interval = record_interval
        self.show_interval   = show_interval
        self.verbose         = verbose

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.record_interval == 0:
            current_losses = np.asarray([logs.get(k) for k in self.loss_keys])
            current_losses[-number_of_dimension:] = 5.0 - current_losses[-number_of_dimension:]
            marginal_epoch_number_list.append(epoch)
            marginal_losses_list.append(current_losses)

            if current_losses[0] < self.previous_loss:
                self.previous_loss = current_losses[0]
                for i, m in enumerate(marginal_model_list):
                    m.save_weights('./best_weights/marginal_model_{}_best_weights.h5'.format(i))

            if self.verbose > 0:
                print('epoch:{}'.format(epoch))
                print([self.loss_keys[i] for i in range(len(self.loss_keys))])
                print(['{:.3f}'.format(current_losses[i]) for i in range(len(self.loss_keys))])

        if epoch % self.show_interval == 0:
            fig, axs = plt.subplots(1, len(marginal_prediction_model_list), figsize=(12, 3))
            for i, mpm in enumerate(marginal_prediction_model_list):
                cdf, pdf, _ = mpm.predict(np.linspace(0, 1, 100).reshape(-1, 1), verbose=0)
                axs[i].hist(X_train[:, i], bins=20, density=True, rwidth=0.7)
                axs[i].plot(np.linspace(0, 1, 100), pdf, 'k-',
                             np.linspace(0, 1, 100), cdf, 'k--')
            plt.tight_layout()
            plt.show()

# Train marginals
# loss_weights: 2 boundary + 2 neg + 2 sum + 2 log  (8 outputs for d=2)
marginal_epoch_number_list = []
marginal_losses_list       = []

marginal_model_train.load_weights('./initial_weights/marginal_model_initial_weights.h5')
marginal_model_train.compile(
    optimizer=keras.optimizers.Nadam(learning_rate=0.001),
    loss="mae",
    loss_weights=[2, 2,   
                  1, 1,   
                  2, 2,   
                  0.1, 0.1])  

start_time = time.time()
marginal_model_train.fit(
    x=marginal_training_input_data,
    y=marginal_training_labels,
    epochs=20000, verbose=0,
    callbacks=[Marginal_Model_Training_Callback(
        record_interval=1000, show_interval=1000, verbose=1)])
print("--- %.1f seconds ---" % (time.time() - start_time))


# COPULA (JOINT) TRAINING DATA
number_boundary_points   = 400
number_partition_per_dim = 50
number_observation_points = 100

# Boundary loss data
joint_boundary_loss_data   = []
joint_boundary_loss_labels = []
for i in range(number_of_dimension):
    temp = np.random.rand(number_boundary_points, number_of_dimension).astype(np.float32)
    temp[:, i] = 0
    joint_boundary_loss_data.append(temp)
    joint_boundary_loss_labels.append(temp[:, i])
for i in range(number_of_dimension):
    temp = np.ones([number_boundary_points, number_of_dimension], dtype=np.float32)
    temp[:, i] = np.random.rand(number_boundary_points).astype(np.float32)
    joint_boundary_loss_data.append(temp)
    joint_boundary_loss_labels.append(temp[:, i])

joint_boundary_loss_data   = np.expand_dims(
    np.concatenate(joint_boundary_loss_data,   axis=0), axis=0)
joint_boundary_loss_labels = np.expand_dims(
    np.concatenate(joint_boundary_loss_labels, axis=0), axis=0)

# Neg/sum loss data 
grids = np.meshgrid(*[np.linspace(0, 1, number_partition_per_dim)
                       for _ in range(number_of_dimension)])
joint_neg_sum_loss_data = np.expand_dims(
    np.column_stack([g.ravel() for g in grids]).astype(np.float32), axis=0)
joint_neg_loss_labels = np.zeros([1, 1], dtype=np.float32)
joint_sum_loss_labels = np.ones( [1, 1], dtype=np.float32)

# Log-likelihood data
joint_log_loss_data   = np.expand_dims(X_train.astype(np.float32), axis=0)
joint_log_loss_labels = np.zeros([1, 1], dtype=np.float32) + 5.0

# Empirical CDF observation data
joint_observation_loss_data = np.expand_dims(
    np.random.rand(number_observation_points, number_of_dimension).astype(np.float32), axis=0)
joint_observation_loss_labels = np.zeros([1, number_observation_points], dtype=np.float32)
for i, crd in enumerate(joint_observation_loss_data[0]):
    flags = X_train[:, 0] <= crd[0]
    for d in range(1, number_of_dimension):
        flags = np.logical_and(flags, X_train[:, d] <= crd[d])
    joint_observation_loss_labels[0, i] = np.sum(flags) / X_train.shape[0]

print('joint_boundary_loss_data:',     joint_boundary_loss_data.shape)
print('joint_neg_sum_loss_data:',      joint_neg_sum_loss_data.shape)
print('joint_log_loss_data:',          joint_log_loss_data.shape)
print('joint_observation_loss_data:',  joint_observation_loss_data.shape)
print('joint_boundary_loss_labels:',   joint_boundary_loss_labels.shape)
print('joint_neg_loss_labels:',        joint_neg_loss_labels.shape)
print('joint_sum_loss_labels:',        joint_sum_loss_labels.shape)
print('joint_log_loss_labels:',        joint_log_loss_labels.shape)
print('joint_observation_loss_labels:', joint_observation_loss_labels.shape)

joint_training_input_data = [joint_boundary_loss_data, joint_neg_sum_loss_data,
                              joint_log_loss_data,      joint_observation_loss_data]
joint_training_labels     = [joint_boundary_loss_labels, joint_neg_loss_labels,
                              joint_sum_loss_labels,      joint_log_loss_labels,
                              joint_observation_loss_labels]

# Build copula training model
joint_boundary_loss_input    = keras.layers.Input(shape=joint_boundary_loss_data.shape[1:])
joint_neg_sum_loss_input     = keras.layers.Input(shape=joint_neg_sum_loss_data.shape[1:])
joint_log_loss_input         = keras.layers.Input(shape=joint_log_loss_data.shape[1:])
joint_observation_loss_input = keras.layers.Input(shape=joint_observation_loss_data.shape[1:])

joint_loss_input_list = [joint_boundary_loss_input, joint_neg_sum_loss_input,
                          joint_log_loss_input,      joint_observation_loss_input]

joint_boundary_loss_output = Lambda(
    lambda x: tf.expand_dims(x, axis=0), name="JB")(
    copula_model(joint_boundary_loss_input[0])[0][:, 0])

_, joint_neg_sum_pdfs, joint_neg_sum_negs = copula_model(joint_neg_sum_loss_input[0])

joint_neg_loss_output = Lambda(
    lambda x: tf.math.reduce_sum(x, keepdims=True) / np.float32(joint_neg_sum_loss_input.shape[1]),
    name="JN")(joint_neg_sum_negs)

joint_sum_loss_output = Lambda(
    lambda x: tf.math.reduce_sum(x, keepdims=True) / np.float32(number_partition_per_dim - 1) ** number_of_dimension,
    name="JS")(joint_neg_sum_pdfs)

joint_log_loss_marginal_cdfs = Lambda(
    lambda x: K.concatenate(x, axis=-1))(
    [marginal_model_list[i](joint_log_loss_input[0, :, i:i+1])[0]
     for i in range(number_of_dimension)])

joint_log_loss_output = Lambda(
    lambda x: tf.math.reduce_sum(tf.math.log(x), keepdims=True) / np.float32(joint_log_loss_input.shape[1]),
    name="JL")(copula_model(joint_log_loss_marginal_cdfs)[1])

joint_observation_marginal_cdfs = Lambda(
    lambda x: K.concatenate(x, axis=-1))(
    [marginal_model_list[i](joint_observation_loss_input[0, :, i:i+1])[0]
     for i in range(number_of_dimension)])

joint_observation_loss_output = Lambda(
    lambda x: tf.expand_dims(x, axis=0), name="JO")(
    copula_model(joint_observation_marginal_cdfs)[0][:, 0])

joint_loss_output_list = [joint_boundary_loss_output, joint_neg_loss_output,
                           joint_sum_loss_output,      joint_log_loss_output,
                           joint_observation_loss_output]

joint_model_train = keras.Model(inputs=joint_loss_input_list,
                                 outputs=joint_loss_output_list)

# Save joint initial weights 
joint_model_train.compile(optimizer=keras.optimizers.Nadam(learning_rate=0.0001),
                           loss="mae")
joint_model_train.fit(x=joint_training_input_data,
                       y=joint_training_labels,
                       epochs=1, verbose=0)
joint_model_train.save_weights('./initial_weights/joint_model_initial_weights.h5')

temp_history   = joint_model_train.fit(x=joint_training_input_data,
                                        y=joint_training_labels,
                                        epochs=1, verbose=0)
joint_loss_keys = list(temp_history.history.keys())
print(joint_loss_keys)

# Callback 
class Joint_Model_Training_Callback(tf.keras.callbacks.Callback):
    def __init__(self, record_interval=10, show_interval=100, verbose=1):
        super().__init__()
        self.loss_keys      = joint_loss_keys
        self.previous_loss  = 9999999
        self.record_interval = record_interval
        self.show_interval   = show_interval
        self.verbose         = verbose

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.record_interval == 0:
            current_losses = np.asarray([logs.get(k) for k in self.loss_keys])
            current_losses[-2] = 5.0 - current_losses[-2]
            joint_epoch_number_list.append(epoch)
            joint_losses_list.append(current_losses)

            if current_losses[0] < self.previous_loss:
                self.previous_loss = current_losses[0]
                copula_model.save_weights('./best_weights/copula_model_best_weights.h5')

            if self.verbose > 0:
                print('epoch:{}'.format(epoch))
                print([self.loss_keys[i] for i in range(len(self.loss_keys))])
                print(['{:.3f}'.format(current_losses[i]) for i in range(len(self.loss_keys))])

        if epoch % self.show_interval == 0:
            domain_space = 50
            x_dom = np.linspace(0, 1, domain_space)
            y_dom = np.linspace(0, 1, domain_space)
            xm, ym = np.meshgrid(x_dom, y_dom)

            Fx, fx, _ = marginal_prediction_model_list[0].predict(
                xm.ravel().reshape(-1, 1), verbose=0)
            Fy, fy, _ = marginal_prediction_model_list[1].predict(
                ym.ravel().reshape(-1, 1), verbose=0)

            uv     = np.column_stack([Fx, Fy]).astype(np.float32)
            c_uv   = copula_prediction_model.predict(uv, verbose=0)[1]
            f_xy   = (c_uv * fx * fy).reshape(domain_space, domain_space)

            fig, ax = plt.subplots(figsize=(8, 6))
            cs = ax.contourf(xm, ym, f_xy, 100)
            fig.colorbar(cs, ax=ax, shrink=0.9)
            ax.set_xlabel('wind_loss (normalised)')
            ax.set_ylabel('flood_loss (normalised)')
            ax.set_title('Joint density f(x,y)')
            plt.tight_layout()
            plt.show()

# Freeze marginals, train copula
for m in marginal_model_list:
    m.trainable = False

joint_epoch_number_list = []
joint_losses_list       = []

joint_model_train.load_weights('./initial_weights/joint_model_initial_weights.h5')
joint_model_train.compile(
    optimizer=keras.optimizers.Nadam(learning_rate=0.001),
    loss="mae",
    loss_weights=[2, 1, 1, 0.1, 5])

start_time = time.time()
joint_model_train.fit(
    x=joint_training_input_data,
    y=joint_training_labels,
    epochs=40000, verbose=0,
    callbacks=[Joint_Model_Training_Callback(
        record_interval=1000, show_interval=1000, verbose=1)])
print("--- %.1f seconds ---" % (time.time() - start_time))


# In[ ]:


domain_space = 80
x_dom = np.linspace(0, 1, domain_space)
y_dom = np.linspace(0, 1, domain_space)
xm, ym = np.meshgrid(x_dom, y_dom)

Fx, fx, _ = marginal_prediction_model_list[0].predict(xm.ravel().reshape(-1,1), verbose=0)
Fy, fy, _ = marginal_prediction_model_list[1].predict(ym.ravel().reshape(-1,1), verbose=0)

uv   = np.column_stack([Fx, Fy]).astype(np.float32)
c_uv = copula_prediction_model.predict(uv, verbose=0)[1]
f_xy = (c_uv * fx * fy).reshape(domain_space, domain_space)

fig, ax = plt.subplots(figsize=(8, 8))
ax.contourf(xm, ym, f_xy, 50, cmap='viridis', alpha=0.7)
ax.scatter(X_test[:,0], X_test[:,1], s=5, c='red', alpha=0.4, label='test data')
ax.set_xlabel('wind_loss'); ax.set_ylabel('flood_loss')
ax.set_title('Fitted joint density vs test data')
ax.legend(); plt.show()


# In[ ]:


n_check = 200
check_pts = np.random.rand(n_check, 2).astype(np.float32)

# Empirical CDF
emp_cdf = np.array([
    np.mean((X_test[:,0] <= pt[0]) & (X_test[:,1] <= pt[1]))
    for pt in check_pts
])

# Model CDF: pass through marginals then copula
u_check = marginal_prediction_model_list[0].predict(check_pts[:,0:1], verbose=0)[0]
v_check = marginal_prediction_model_list[1].predict(check_pts[:,1:2], verbose=0)[0]
uv_check = np.column_stack([u_check, v_check]).astype(np.float32)
model_cdf = copula_prediction_model.predict(uv_check, verbose=0)[0].ravel()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(emp_cdf, model_cdf, s=10, alpha=0.5)
ax.plot([0,1],[0,1],'r--', label='perfect fit')
ax.set_xlabel('Empirical CDF'); ax.set_ylabel('Model CDF')
ax.set_title('Empirical vs Fitted joint CDF'); ax.legend()
plt.show()

# Quantify with RMSE
rmse = np.sqrt(np.mean((emp_cdf - model_cdf)**2))
print(f"CDF RMSE: {rmse:.4f}  (lower is better, 0 = perfect)")

