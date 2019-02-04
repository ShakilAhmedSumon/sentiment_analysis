from curriculam_preprocess import *
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
from keras import regularizers
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
#K.tensorflow_backend._get_available_gpus()


def curriculam_training(min_len, max_len):

	x_train, x_test, y_train, y_test=curriculam_text_target(min_len,max_len)

	x_train = x_train.reshape(x_train.shape[0], max_len, 1)
	x_test =  x_test.reshape(x_test.shape[0], max_len, 1)


	rnn_size=300
	fc_size=150
	output_dim=3


	input_shape=(max_len,1)

	init = random_normal(stddev=0.046875)

	input_data = Input(name='the_input', shape=input_shape, dtype='float32')
	q = TimeDistributed(Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation='relu',	kernel_regularizer=regularizers.l2(0.01)))(input_data)
	#q = TimeDistributed(Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation='relu'))(q)
	#q = TimeDistributed(Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation='relu'))(q)
	q = TimeDistributed(Dense(fc_size, name='fc1', kernel_initializer=init, bias_initializer=init, activation='relu', kernel_regularizer=regularizers.l2(0.01)))(q)
	q = TimeDistributed(Dense(fc_size, name='fc2', kernel_initializer=init, bias_initializer=init, activation='relu'))(q)
	q = TimeDistributed(Dense(fc_size, name='fc3', kernel_initializer=init, bias_initializer=init, activation='relu'))(q)
	'''	
	q = Bidirectional(LSTM(rnn_size, return_sequences=True, activation='relu',
                                kernel_initializer='he_normal', name='birnn'), merge_mode='sum')(q)
	q = Bidirectional(LSTM(rnn_size, return_sequences=True, activation='relu',
                                kernel_initializer='he_normal', name='birnn'), merge_mode='sum')(q)
	'''	
	q=Flatten()(q)
	
	y_pred = Dense(output_dim, name="y_pred", kernel_initializer=init, bias_initializer=init, activation="softmax")(q)


	model=Model(inputs=input_data, outputs=y_pred)

	model.summary()
	#if max_len>20:
		#model.load_weights(str(min_len-20)+"_"+str(max_len-20)+".h5")

		
	model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])

	model.fit(x_train, y_train, batch_size=100, epochs=500, verbose=1,validation_data=(x_test,y_test), 
	                        callbacks=[keras.callbacks.TensorBoard(log_dir="logs/sen1/{}".format(time()), 
		                    histogram_freq=0, write_graph=True, write_images=True)]
         )
	model.save_weights(str(min_len)+"_"+str(max_len)+".h5")

	


def curriculam_train():
	i=0

	min_len=1
	max_len=20

	for i in range(20):
		curriculam_training(min_len,max_len)
		#min_len=min_len+20
		max_len=max_len+20



if __name__ == '__main__':
	curriculam_training(1,160)
	
	


