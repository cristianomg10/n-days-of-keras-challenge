"""
Day 4 - Testing real life classification
"""

import keras
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils as utils
from sklearn.preprocessing import LabelEncoder

from random import shuffle
import numpy as np

import csv

# Reproducibility
seed = 7
np.random.seed(seed)

# Loading data
dataset = []
x_train = []
y_train = []
x_test = []
y_test = []
remove_header = True
i = 0

with open('train-test-logan.csv', newline='') as csvfile:
    file_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in file_reader:
        if not(i == 0 and remove_header):
            dataset.append(row)
        i = i + 1

data = []
target = []
for i in dataset:
    data.append(i[0: len(i) - 1])
    target.append(i[len(i) - 1])

# Fix classes
le = LabelEncoder()
target = le.fit_transform(target)

# Shuffle
c = list(zip(data, target))
shuffle(c)
data, target = zip(*c)

# Split
numTotal = len(data)
percent = round(numTotal * 0.8)

dataTrain = data[0:percent]
targetTrain = target[0:percent]
dataTest = data[percent+1:]
targetTest = target[percent+1:]

# Design the MLP
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=16))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Train the MLP
model.fit(np.asarray(dataTrain),
          np.asarray(targetTrain),
          epochs=100,
          verbose=1,
          batch_size=10)

# Apply test on the model
score = model.evaluate(np.asarray(dataTest), np.asarray(targetTest))
print(score)

