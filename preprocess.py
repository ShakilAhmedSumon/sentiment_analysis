import numpy as np 
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

DATA_PATH = "sentiment_analysis.xlsx"


def read_file(DATA_PATH):
	'''
	reads the file provided:
	DATA_PATH: The path of the file.

	Returns:
	y          : The target vector of the dataset.
	bangla_text: The text of the comments.
	'''

	df=pd.read_excel(DATA_PATH)
	feature_column_name=['text']
	predicted_class_name=['score']

	y=df[predicted_class_name].values
	bangla_text=df[feature_column_name].values

	return y, bangla_text


def char_to_utf(bangla_text):
	'''
	function: Converts bangla characters to it's corresponding utf numbers.

	bangla_text: Text of the comments;

	Returns:
	utf_all: Numpy array of the utf numbers.
	'''

	utf_all=[]

	for text in bangla_text:
		for string in text:
			utf=[]
			for char in string:
				number=ord(char)
				utf.append(number)
			utf=np.asarray(utf)
			#
			if utf.shape[0]>100:
				utf=utf[:100]
			else:
				pad_width=100-utf.shape[0]
				utf=np.pad(utf,pad_width=(0,pad_width),mode='constant')

			#
			utf_all.append(utf)
	utf_all=np.asarray(utf_all)

	return utf_all



def maximum_length(utf_all):
	'''
	utf_all: The numpy array which holds the utf of all the characters.

	Returns: The maximum length of the comments in the dataset.
	'''
	length=0
	for utf in utf_all:
		shape= utf.shape[0]
		if length<shape:
			length=shape
	return length 



def padding(utf_all,max_len):
	'''
	utf_all: The numpy array which holds the utf of all the characters.
	max_len: The maximum length we want to take from a sentence.

	Returns: The padded array of the sequence of utf numbers.
	'''
	for utf in utf_all:
		if utf.shape[0]>max_len:
			utf=utf[:max_len]
			#print("Cut: ", utf)
		else:
			pad_width=max_len-utf.shape[0]
			utf=np.pad(utf,pad_width=(0,pad_width),mode='constant')
			#print("Padded: ", utf)

	np.save("utf_all.npy", utf_all)
	#print(utf_all)
	return utf_all 


def train_test_set():

	'''
	Return x_train, x_test, y_train, y_test

	'''

	y, bangla_text=read_file(DATA_PATH)
	utf_all=char_to_utf(bangla_text)
	print(utf_all.shape)
	x= padding(utf_all,100)

	y=np.asarray(y)
	y=to_categorical(y)

	#print(y.shape)
	#print(x.shape)
	#print(x)


	x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)

	return x_train, x_test, y_train, y_test





if __name__ == '__main__':
	x_train, x_test, y_train, y_test=train_test_set()

	

	



