#!/usr/bin/env python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_folder = 'decisions_sub/'
dataset_filename = data_folder + 'datasets.npz'
label_filename = data_folder + 'label_mapping.sav'
model_filename = data_folder + 'model.sav'


try:
      data = np.load(dataset_filename)
      label_mapping = pickle.load(open(label_filename, 'rb'))
except IOError:
      print('File not found. Bye.')
      exit()


x_train, y_train, x_test, y_test = [data[key] for key in ['x_train', 'y_train', 'x_test', 'y_test']]

print(f'\n\nShapes: x_train = {x_train.shape}, y_train = {y_train.shape}, '
      f'x_test = {x_test.shape}, y_test = {y_test.shape}\n\n', flush = True)


# Instantiate Model:
# model = KNeighborsClassifier(n_neighbors = 5)
# model = RandomForestClassifier(n_estimators = 50, verbose = 2) 
# model = GradientBoostingClassifier(n_estimators = 50, random_state = 42)
# model = LinearSVC()
# model = SVC(gamma='auto')
model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = GaussianNB()

# Train:
model.fit(x_train, y_train)

# Model accuracy etc.:
y_pred = model.predict(x_test)
print('Test set accuracy:\n' + classification_report(y_test, y_pred, target_names = label_mapping), flush = True)

# Model cross-validation:
X = np.vstack((x_train, x_test))
Y = np.append(y_train, y_test, axis = 0)
X, Y = shuffle(X, Y)

score = cross_val_score(model, X, Y, cv = 10)
print(f'\nCross-validation results:\n{score}\n\nMean: {np.mean(score)}', flush = True)


# Confusion matrix:
cm = confusion_matrix(y_test, y_pred, labels=range(len(label_mapping)))
plot_confusion_matrix(model, x_test, y_test, display_labels=label_mapping)
plt.xticks(rotation = -90)
plt.tight_layout()
plt.show()

# # Save the model to disk
pickle.dump(model, open(model_filename, 'wb'))