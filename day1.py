"""
Day 1

MLP multi-class classification.

"""

import keras
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD

from sklearn import datasets
from random import shuffle

import numpy as np

model = Sequential()
model.add(Dense(64, activation='sigmoid', input_dim=4))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Prepare the data
iris = datasets.load_iris()
data = iris.data
target = iris.target

c = list(zip(data, target))
shuffle(c)
data, target = zip(*c)

numTotal = len(data)
percent = round(numTotal * 0.8)

dataTrain = data[0:percent]
targetTrain = target[0:percent]
dataTest = data[percent+1:]
targetTest = target[percent+1:]

model.fit(np.asarray(dataTrain), np.asarray(targetTrain), epochs=10000, verbose=1)

score = model.evaluate(np.asarray(dataTest), np.asarray(targetTest))
print(score)

print(model.get_config())
print(model.predict_classes(np.asarray(data[0:20])))

# print errors
print(sum(abs(model.predict_classes(np.asarray(data[0:20])) - target[0:20])))