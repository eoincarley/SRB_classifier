import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers


model = models.Sequential()
model.add(layers.Dense(100, activation = "relu", input_shape=(7100, 1)))
model.add(layers.Dense(3, activation = "softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])