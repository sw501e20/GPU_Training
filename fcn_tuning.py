#!/usr/bin/env python
import keras
import numpy as np
import datetime as dt
import os
from kerastuner.tuners import RandomSearch
import pickle
import sys

dataset_file_name = 'datasets.npz'
project_name = 'tuner'

if not os.path.isfile(dataset_file_name):
      print('Dataset file not found. Bye.')
      exit()

logdir = 'logs/'
os.makedirs(logdir, exist_ok = True)

data = np.load(dataset_file_name)
x_train, y_train, x_test, y_test = [data[key] for key in ['x_train', 'y_train', 'x_test', 'y_test']]

print(f'\n\nShapes: x_train = {x_train.shape}, y_train = {y_train.shape}, '
      f'x_test = {x_test.shape}, y_test = {y_test.shape}\n\n')


# Define hyperparameters:
input_shape = x_train.shape[1:]
num_classes = y_train.shape[1]
mini_batch_size = 128
num_epochs = 50

def build_model(hp):

      input_layer = keras.layers.Input(input_shape)

      conv1 = keras.layers.Conv1D(filters = hp.Int('units_1',
                                            min_value=32,
                                            max_value=512,
                                            step=32), 
                                  kernel_size = hp.Int('kernel_1',
                                            min_value=1,
                                            max_value=10,
                                            step=2), padding = 'same')(input_layer)
      conv1 = keras.layers.BatchNormalization()(conv1)
      conv1 = keras.layers.ReLU()(conv1)

      conv2 = keras.layers.Conv1D(filters = hp.Int('units_2',
                                            min_value=32,
                                            max_value=512,
                                            step=32), 
                                  kernel_size = hp.Int('kernel_2',
                                            min_value=1,
                                            max_value=10,
                                            step=2), padding = 'same')(conv1)
      conv2 = keras.layers.BatchNormalization()(conv2)
      conv2 = keras.layers.ReLU()(conv2)

      conv3 = keras.layers.Conv1D(filters = hp.Int('units_3',
                                            min_value=32,
                                            max_value=512,
                                            step=32), 
                                  kernel_size = hp.Int('kernel_3',
                                            min_value=1,
                                            max_value=10,
                                            step=2), padding = 'same')(conv2)
      conv3 = keras.layers.BatchNormalization()(conv3)
      conv3 = keras.layers.ReLU()(conv3)

      gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

      output_layer = keras.layers.Dense(num_classes, activation = 'softmax')(gap_layer)

      model = keras.models.Model(inputs = input_layer, outputs = output_layer)

      model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])

      # model.summary()

      return model


# Declare tuner:
tuner = RandomSearch(build_model,
                     objective = 'val_accuracy',
                     max_trials = 1000,
                     executions_per_trial = 3,
                     directory = logdir,
                     project_name = project_name)


tuner.search_space_summary()

# Define callbacks:
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 50, min_lr = 0.0001)
callbacks = [reduce_lr]

# Train model:
tuner.search(x_train, y_train, batch_size = mini_batch_size, epochs = num_epochs,
		 verbose = 1, validation_data = (x_test, y_test), callbacks = callbacks)


sys.stdout = open(logdir + project_name + '/results.txt', 'w')
tuner.results_summary(num_trials = 100)

model = tuner.get_best_models(num_models = 1)[0]
model.save(logdir + project_name + '/best_model.hdf5')