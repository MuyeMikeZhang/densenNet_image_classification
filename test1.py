
import keras
from keras import layers

model = keras.Sequential()
model.add(layers.Dense(1, input_dim=1))
model.summary()