#!/usr/bin/env python
import keras
import numpy as np
import datetime as dt
import os
from kerastuner.tuners import RandomSearch
import pickle
import sys

dataset_file_name = 'mlp tuning/datasets.npz'
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
num_epochs = 100

def build_model(hp):

    #   input_layer = keras.layers.Input(input_shape)

    #   conv1 = keras.layers.Conv1D(filters = 48, kernel_size = 1, padding = 'same')(input_layer)
    #   conv1 = keras.layers.BatchNormalization()(conv1)
    #   conv1 = keras.layers.ReLU()(conv1)
    #   conv1 = keras.layers.Dropout(hp.Float('dropout_1', min_value=0.1, max_value=1, step=0.4))(conv1)


    #   conv2 = keras.layers.Conv1D(filters = 48, kernel_size = 1, padding = 'same')(conv1)
    #   conv2 = keras.layers.BatchNormalization()(conv2)
    #   conv2 = keras.layers.ReLU()(conv2)
    #   conv2 = keras.layers.Dropout(hp.Float('dropout_2', min_value=0.1, max_value=1, step=0.4))(conv2)


    #   conv3 = keras.layers.Conv1D(filters = 40, kernel_size = 1, padding = 'same')(conv2)
    #   conv3 = keras.layers.BatchNormalization()(conv3)
    #   conv3 = keras.layers.ReLU()(conv3)
    #   conv3 = keras.layers.Dropout(hp.Float('dropout_3', min_value=0.1, max_value=1, step=0.4))(conv3)


    #   gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    #   output_layer = keras.layers.Dense(num_classes, activation = 'softmax')(gap_layer)

    #   model = keras.models.Model(inputs = input_layer, outputs = output_layer)
    input_layer = keras.layers.Input(input_shape)

    # flatten/reshape because when multivariate all should be on the same axis 
    input_layer_flattened = keras.layers.Flatten()(input_layer)

    #layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
    layer_1 = keras.layers.Dense(hp.Int('units_1',
                                            min_value=32,
                                            max_value=65,
                                            step=8), activation='relu')(input_layer_flattened)

    #layer_2 = keras.layers.Dropout(0.2)(layer_1)
    layer_2 = keras.layers.Dense(hp.Int('units_2',
                                            min_value=32,
                                            max_value=65,
                                            step=8), activation='relu')(layer_1)

    #layer_3 = keras.layers.Dropout(0.2)(layer_2)
    layer_3 = keras.layers.Dense(32, activation='relu')(layer_2)

    #output_layer = keras.layers.Dropout(0.3)(layer_3)
    output_layer = keras.layers.Dense(num_classes, activation='softmax')(layer_3)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

    # model.summary()

    return model


# Declare tuner:
tuner = RandomSearch(build_model,
                     objective = 'val_accuracy',
                     max_trials = 500,
                     executions_per_trial = 2,
                     directory = logdir,
                     project_name = project_name)


tuner.search_space_summary()

# Define callbacks:
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)
callbacks = [reduce_lr]

# Train model:
tuner.search(x_train, y_train, batch_size = mini_batch_size, epochs = num_epochs,
		 verbose = 1, validation_data = (x_test, y_test), callbacks = callbacks)


sys.stdout = open(logdir + project_name + '/results.txt', 'w')
tuner.results_summary(num_trials = 500)

model = tuner.get_best_models(num_models = 1)[0]
model.save(logdir + project_name + '/best_model.hdf5')