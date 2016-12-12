from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, \
  Activation, Input, merge, Convolution2D, MaxPooling2D
from keras.optimizers import SGD, adam
import numpy as np

from keras.utils import np_utils
from keras.datasets import cifar10, mnist
from ppap.layers.ppconv import PPConv
from ppap.layers.HyperNeat import HyperNeat

if False:
  rows, cols, channels = 28, 28, 1
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  inputs = Input(shape=(1, 28, 28), name="inputs")
  X_train = np.expand_dims(X_train, axis=1)
  X_test = np.expand_dims(X_test, axis=1)
else:
  rows, cols, channels = 32, 32, 3
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()
  inputs = Input(shape=(3, 32, 32), name="inputs")

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255.0
X_test /= 255.0
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


conv_1 = Convolution2D(16, 3, 3, activation="relu") (inputs)
max_pool_1 = MaxPooling2D()(conv_1)
# conv_2 = Convolution2D(32, 3, 3, activation="relu") (max_pool_1)
# max_pool_2 = MaxPooling2D()(conv_2)
# dropout_1 = Dropout(0.25)(max_pool_2)

# ppconv_1 = PPConv(weight_shape=(3, 3), layer_sizes=[4, 4, 4],
#                   nb_filters=33, border_mode="valid")(inputs)
# ppconv_1 =  Activation(activation="relu")(ppconv_1)
# max_pool_1 = MaxPooling2D()(ppconv_1)
# dropout_1 = Dropout(p=0.5)(max_pool_1)

ppconv_2 = HyperNeat(weight_shape=(3, 3), hidden_dim=20,
                  nb_filters=33, border_mode="valid")(max_pool_1)
ppconv_2 = Activation(activation="relu")(ppconv_2)
max_pool_2 = MaxPooling2D()(ppconv_2)
# dropout_2 = Dropout(p=0.5)(max_pool_2)


flat = Flatten()(max_pool_2)
dense_1 = Dense(output_dim=128, activation="relu")(flat)
dropout_3 = Dropout(0.25)(dense_1)

dense_2 = Dense(output_dim=10, activation="softmax")(dropout_3)

model = Model(input=inputs, output=[dense_2])
sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=[
  "accuracy"])

model.summary()
print (X_train.shape)
model.fit(x=X_train, y=y_train, nb_epoch=20, validation_data=[X_test,
                                                              y_test],
          verbose=1, batch_size=256)

"""
DUMP
"""
# ppconv_1 = PPConv(weight_shape=(3, 3), layer_sizes=[4, 4, 4],
#                   nb_filters=33, border_mode="valid")(max_pool1)
# ppconv_1 =  Activation(activation="relu")(ppconv_1)
# max_pool2 = MaxPooling2D()(ppconv_1)
# ppconv_1 = Dropout(p=0.5)(ppconv_1)

# ppconv_1 = PPConv(weight_shape=(3, 3), layer_sizes=[50, 40, 10],
#                   nb_filters=22, border_mode="valid")(ppconv_1)
# ppconv_1 =  Activation(activation="relu")(ppconv_1)

# ppconv_2 = PPConv(weight_shape=(3, 3), layer_sizes=[50, 40, 10],
#                   nb_filters=33)(ppconv_1)
# ppconv_2 =  Activation(activation="relu")(ppconv_2)
# ppconv_2 = Dropout(p=0.5)(ppconv_2)