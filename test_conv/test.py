from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, merge
from keras.optimizers import SGD, adam
import numpy as np

from keras.utils import np_utils
from keras.datasets import mnist
from ppap.layers.ppconv import PPConv

img_rows, img_cols = 28, 28
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=1)
y_train = np_utils.to_categorical(y_train, 10)
X_test = np.expand_dims(X_test, axis=1)
y_test = np_utils.to_categorical(y_test, 10)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

inputs = Input(shape=(1, 28, 28), name="inputs")

ppconv_1 = PPConv(weight_shape=(3, 3), layer_sizes=[50, 40, 10],
                  nb_filters=9)(inputs)
ppconv_1 =  Activation(activation="relu")(ppconv_1)
# ppconv_1 = Dropout(p=0.5)(ppconv_1)

ppconv_2 = PPConv(weight_shape=(3, 3), layer_sizes=[50, 40, 10],
                  nb_filters=33)(ppconv_1)
ppconv_2 =  Activation(activation="relu")(ppconv_2)
# ppconv_2 = Dropout(p=0.5)(ppconv_2)

flat = Flatten()(ppconv_2)
# dense_1 = Dense(output_dim=128, activation="relu")(flat)
dense_2 = Dense(output_dim=10, activation="softmax")(flat)

model = Model(input=inputs, output=[dense_2])
sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[
  "accuracy"])

model.summary()
print (X_train.shape)
model.fit(x=X_train, y=y_train, nb_epoch=10, validation_data=[X_test, y_test], verbose=1, batch_size=32)
