"""
Day 2 - Regression

Prediction using Boston Housing dataset.
"""

# Imports
import keras
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.datasets import boston_housing

import numpy as np

# Reproducibility
seed = 7
np.random.seed(seed)

# Regressor
model = Sequential();
model.add(Dense(8, activation='relu', input_dim=13))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

# Get the data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Train the model
model.fit(x=x_train, y=y_train, epochs=5000, shuffle=True)

# Test the model
print ('-------------- Test --------------')
score = model.evaluate(x=x_test,y=y_test)
print (score)