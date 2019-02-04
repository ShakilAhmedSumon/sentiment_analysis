from preprocess import *
import keras
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import *
from keras import layers
from keras.callbacks import TensorBoard
from time import time
from keras.models import Model
import os
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Lambda
from keras.layers import Dropout
from keras.regularizers import l2
from keras.initializers import random_normal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from time import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#K.tensorflow_backend._get_available_gpus()



x_train, x_test, y_train, y_test=train_test_set()

x_train = x_train.reshape(x_train.shape[0], 100, 1)
x_test =  x_test.reshape(x_test.shape[0], 100, 1)


rnn_size=50
fc_size=100
output_dim=3


input_shape=(100,1)

init = random_normal(stddev=0.046875)

input_data = Input(name='the_input', shape=input_shape, dtype='float32')
q = TimeDistributed(Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation='relu'))(input_data)
q = TimeDistributed(Dense(fc_size, name='fc2', kernel_initializer=init, bias_initializer=init, activation='relu'))(q)
q = TimeDistributed(Dense(fc_size, name='fc3', kernel_initializer=init, bias_initializer=init, activation='relu'))(q)
q = Bidirectional(LSTM(rnn_size, return_sequences=True, activation='relu',
                                kernel_initializer='he_normal', name='birnn'), merge_mode='sum')(q)
q=Flatten()(q)
y_pred = Dense(output_dim, name="y_pred", kernel_initializer=init, bias_initializer=init, activation="softmax")(q)


model=Model(inputs=input_data, outputs=y_pred)

model.summary()

model.compile(loss='mse',optimizer='adadelta',metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=10, verbose=1,validation_data=(x_test,y_test), 
	                        callbacks=[keras.callbacks.TensorBoard(log_dir="logs/sen1/{}".format(time()), 
		                    histogram_freq=0, write_graph=True, write_images=True)]
         )


score,acc=model.evaluate(x_test, y_test, batch_size=100)
pred=model.predict(x_test)
pred=np.argmax(pred,axis=1)
y_test_hot=np.argmax(y_test,axis=1)
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_test_hot,pred, target_names=target_names))
print(confusion_matrix(y_test_hot,pred))