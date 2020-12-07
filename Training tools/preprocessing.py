#!/usr/bin/env python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import pickle
from math import pi, sin
import os


deep = False
num_timesteps = 1
aggregation = 100

# Input files:
csv_filename = 'armband_data.csv'

# Output files:
output_folder = 'fcn_sub_decisions/'
dataset_filename = 'datasets.npz'
scaler_filename = 'scaler.sav'
label_filename = 'label_mapping.sav'


df = pd.read_csv(csv_filename, skipinitialspace=True)
#df = df[df['GESTURE'].isin(['left turn', 'right turn'])]

df.sort_values('TIME', inplace=True)

df['GESTURE'], label_mapping = pd.factorize(df['GESTURE'])

sine_transform = lambda angle: abs(sin(angle * pi / 180))
df['EULER_ANGLE_X'] = df['EULER_ANGLE_X'].apply(sine_transform) 
df['EULER_ANGLE_Y'] = df['EULER_ANGLE_Y'].apply(sine_transform) 
df['EULER_ANGLE_Z'] = df['EULER_ANGLE_Z'].apply(sine_transform) 

sequences = df.groupby(['SUBJECTID', 'REPID'])
sequences = [sequences.get_group(group).drop(['TIME'], axis=1) for group in sequences.groups]


prune = lambda num_rows: num_rows - num_rows % (num_timesteps * aggregation)
sequences = [sequence[:prune(sequence.shape[0])] for sequence in sequences]

timeseries, gestures, subjects = zip(*[(sequence.drop(['SUBJECTID', 'REPID', 'GESTURE'], axis=1),
                                           sequence['GESTURE'].iat[0],
                                           sequence['SUBJECTID'].iat[0]) for sequence in sequences])

timeseries = [ts.fillna(ts.mean()) for ts in timeseries]

for ts in timeseries:
    ts.reset_index(inplace=True, drop=True)

timeseries = [ts.groupby(ts.index // aggregation).mean().values for ts in timeseries]

encoded_labels = to_categorical(gestures) if deep else gestures


# X, Y = [], []
# if deep:
#     for i in range(len(timeseries)):
#         split = timeseries[i].shape[0] // num_timesteps // aggregation
#         for j in range(split):
#             X.append(timeseries[i][j * num_timesteps:(j + 1) * num_timesteps]); Y.append(encoded_labels[i])
# else:
#     for i in range(len(timeseries)):
#         split = timeseries[i].shape[0]
#         for j in range(split):
#             X.append(timeseries[i][j]); Y.append(encoded_labels[i])
# x_train, x_test, y_train, y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2, random_state=42)

x_train, x_test, y_train, y_test = [], [], [], []
if deep:
    for i in range(len(timeseries)):
        if subjects[i] == 5:
            split = timeseries[i].shape[0] // num_timesteps // aggregation
            for j in range(split):
                x_test.append(timeseries[i][j * num_timesteps:(j + 1) * num_timesteps]); y_test.append(encoded_labels[i])
        else:
            split = timeseries[i].shape[0] // num_timesteps // aggregation
            for j in range(split):
                x_train.append(timeseries[i][j * num_timesteps:(j + 1) * num_timesteps]); y_train.append(encoded_labels[i])
else:
    for i in range(len(timeseries)):
        split = timeseries[i].shape[0]
        if subjects[i] == 5:
            for j in range(split):
                x_test.append(timeseries[i][j]); y_test.append(encoded_labels[i])
        else:
            for j in range(split):
                x_train.append(timeseries[i][j]); y_train.append(encoded_labels[i])
x_train, x_test, y_train, y_test = [np.array(lst) for lst in [x_train, x_test, y_train, y_test]]

# Scale/normalize data:
scaler = StandardScaler()

if deep:
    batch_size, num_timesteps, num_features = x_train.shape
    x_train = np.reshape(x_train, (-1, num_features))
    x_train = scaler.fit_transform(x_train)
    x_train = np.reshape(x_train, (batch_size, num_timesteps, num_features))

    batch_size, num_timesteps, num_features = x_test.shape
    x_test = np.reshape(x_test, (-1, num_features))
    x_test = scaler.transform(x_test)
    x_test = np.reshape(x_test, (batch_size, num_timesteps, num_features))
else:
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)


print(f'\n\nShapes: x_train = {x_train.shape}, y_train = {y_train.shape}, '
      f'x_test = {x_test.shape}, y_test = {y_test.shape}\n\n')


# Save data sets:
if output_folder:
    os.mkdir(output_folder)

np.savez(output_folder + dataset_filename, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
pickle.dump(scaler, open(output_folder + scaler_filename, 'wb'))
pickle.dump(label_mapping, open(output_folder + label_filename, 'wb'))