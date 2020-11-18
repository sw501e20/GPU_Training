#!/usr/bin/env python
import keras
import numpy as np
import datetime as dt
import os

dataset_filename = 'datasets.npz'

try:
      data = np.load(dataset_filename)
except:
      print('Dataset file not found. Bye.')
      exit()

logdir = 'logs/' + dt.datetime.now().strftime("%m-%d-%H%M")
os.makedirs(logdir, exist_ok = True)
os.system(f'START "" tensorboard --logdir={logdir}')

x_train, y_train, x_test, y_test = [data[key] for key in ['x_train', 'y_train', 'x_test', 'y_test']]

print(f'\n\nShapes: x_train = {x_train.shape}, y_train = {y_train.shape}, '
      f'x_test = {x_test.shape}, y_test = {y_test.shape}\n\n')


# Define hyperparameters:
input_shape = x_train.shape[1:]
num_classes = y_train.shape[1]
batch_size = 128 
num_epochs = 2000


# Define model:
input_layer = keras.layers.Input(input_shape)

conv1 = keras.layers.Conv1D(filters = 48, kernel_size = 1, padding = 'same')(input_layer)
conv1 = keras.layers.BatchNormalization()(conv1)
conv1 = keras.layers.ReLU()(conv1)

conv2 = keras.layers.Conv1D(filters = 48, kernel_size = 1, padding = 'same')(conv1)
conv2 = keras.layers.BatchNormalization()(conv2)
conv2 = keras.layers.ReLU()(conv2)

conv3 = keras.layers.Conv1D(filters = 40, kernel_size = 1, padding = 'same')(conv2)
conv3 = keras.layers.BatchNormalization()(conv3)
conv3 = keras.layers.ReLU()(conv3)

gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

output_layer = keras.layers.Dense(num_classes, activation = 'softmax')(gap_layer)

model = keras.models.Model(inputs = input_layer, outputs = output_layer)

model.summary()

# Compile model and define callbacks:
model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics = ['accuracy'])

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 50, min_lr = 0.0001)
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath = logdir + '/best_model.hdf5', monitor = 'loss', save_best_only = True)
tensorboard = keras.callbacks.TensorBoard(log_dir = logdir)

callbacks = [reduce_lr, model_checkpoint, tensorboard]


# Train model:
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = num_epochs,
			        verbose = 1, validation_data = (x_test, y_test), callbacks = callbacks)

model.save(logdir + '/last_model.hdf5')


# Evaluate model: 
score = model.evaluate(x_test, y_test, verbose = 0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')