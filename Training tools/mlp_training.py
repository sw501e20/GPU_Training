#!/usr/bin/env python
import keras
import numpy as np
import datetime as dt
import os

dataset_filename = 'fcn dropout tuning/fcn_sub5_dropout/datasets.npz'

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
num_epochs = 250


# Define model:
input_layer = keras.layers.Input(input_shape)

# flatten/reshape because when multivariate all should be on the same axis 
input_layer_flattened = keras.layers.Flatten()(input_layer)

#layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
layer_1 = keras.layers.Dense(416, activation='relu')(input_layer_flattened)

#layer_2 = keras.layers.Dropout(0.2)(layer_1)
layer_2 = keras.layers.Dense(480, activation='relu')(layer_1)

#layer_3 = keras.layers.Dropout(0.2)(layer_2)
layer_3 = keras.layers.Dense(224, activation='relu')(layer_2)

#output_layer = keras.layers.Dropout(0.3)(layer_3)
output_layer = keras.layers.Dense(num_classes, activation='softmax')(layer_3)

model = keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'])

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=logdir + '/best_model.hdf5', monitor='loss', 
    save_best_only=True)

tensorboard = keras.callbacks.TensorBoard(log_dir = logdir)
callbacks = [reduce_lr,model_checkpoint, tensorboard]

model.summary()

# Train model:
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = num_epochs,
			        verbose = 1, validation_data = (x_test, y_test), callbacks = callbacks)

model.save(logdir + '/last_model.hdf5')


# Evaluate model: 
score = model.evaluate(x_test, y_test, verbose = 0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')