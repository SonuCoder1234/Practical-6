from tensorflow import keras

from keras.layers import Dense, Dropout,Flatten

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import pandas as pd

(x_train, y_train),(x_test,y_test) = keras.datasets.cifar10.load_data()

x_train = x_train[:5000,:]

y_train = y_train[:5000,:]

x_test = x_test[:1000]

y_test = y_test[:1000]

x_test.shape

x_train = x_train / 255

x_test = x_test / 255

from keras.utils import to_categorical

y_train = to_categorical(y_train , 10)

y_test = to_categorical(y_test,10)

y_train.shape

basemodel = 

keras.applications.VGG16(weights='imagenet',include_top=False,input_shape=(32,32,3))

for layer in basemodel.layers:

layer.trainable = False

x = Flatten()(basemodel.output)

x = Dense(256,activation='relu')(x)

x = Dropout(0.3)(x)

x = Dense(128,activation='relu')(x)

x = Dropout(0.3)(x)

x = Dense(10,activation='softmax')(x)

model = keras.models.Model(inputs = basemodel.input , outputs = x)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train,y_train, epochs = 10,batch_size = 64,validation_data=(x_test,y_test))

basemodel = 

keras.applications.VGG16(weights='imagenet',include_top=False,input_shape=(32,32,3))

for layer in basemodel.layers:

layer.trainable = False

for layer in basemodel.layers[len(basemodel.layers)-4:]:

layer.trainable = True

x = Flatten()(basemodel.output)

x = Dense(512,activation='relu')(x)

x = Dropout(0.3)(x)

x = Dense(256,activation='relu')(x)

x = Dropout(0.3)(x)

x = Dense(10,activation='softmax')(x)

model = keras.models.Model(inputs=basemodel.input, outputs = x)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentrop

y',metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

model.evaluate(x_test,y_test)
