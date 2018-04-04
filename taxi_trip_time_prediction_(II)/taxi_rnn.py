# --------------------------------------
# Library
# --------------------------------------
# Tools
from pandas import read_csv
import numpy
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler

# Model
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers

# Validation tool
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

########################
#---- Load dataset ----#
########################
dataset = read_csv('reordered.csv')
print (dataset.head(5))
values  = dataset.values

########################
#------- dataset ------#
########################
n_data = len(dataset.index) # total number of data
train_fields = ['HOUR', 'CALL_TYPE', 'TAXI_ID']
train = read_csv('reordered.csv', usecols=train_fields)
label_fields = ['POLYLINE']
label = read_csv('reordered.csv', usecols=label_fields)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train)
scaled_label = scaler.fit_transform(label)
print "scaled_train"
print (scaled_train)
print "\n"
print "scaled_label"
print (scaled_label)

# split into train and test setsls
x_train, x_test, y_train, y_test = train_test_split(
									scaled_train, 
									scaled_label, 
									test_size=0.3, 
									random_state=42)

# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# design network
model = Sequential()
model.add(LSTM(500, return_sequences = True, input_shape=(x_train.shape[1], x_train.shape[2])))
#model.add(LSTM(500, return_sequences = True))
model.add(LSTM(500))
model.add(Dense(1)) # output node = 1
#model.compile(loss='mae', optimizer='adam')
# optimizers parameter to control gradient clipping
sgd = optimizers.SGD(lr=1, clipvalue=0.5) # clipping, range from -0.5 ~ 0.5
model.compile(loss='mse', optimizer=sgd)
model.summary()
# fit network
history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), verbose=2, shuffle=False)

# make a prediction
yhat = model.predict(x_test)

# calculate RMSE
rmse = sqrt(mean_squared_error(y_test, yhat))
print('Test RMSE: %.3f' % rmse)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
