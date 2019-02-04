import pandas as pd
import numpy as np 
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

DATA_PATH= "sentiment_analysis.xlsx"

def read_file(DATA_PATH):
	df = pd.read_excel(DATA_PATH)
	feature_column_name = ['text']
	predicted_class_name = ['score']
	

	y = df[predicted_class_name].values
	bangla_text = df[feature_column_name].values

	return y, bangla_text


def char_to_utf(bangla_text, max_len):
	utf_all = []

	for text in bangla_text:
		for string in text:
			utf = []
			for char in string:
				number = ord(char)
				utf.append(number)
			utf = np.asarray(utf)
			if(utf.shape[0]>max_len):
				utf = utf[:max_len]
			else:
				pad_width = max_len - utf.shape[0]
				utf=np.pad(utf,pad_width=(0,pad_width),mode='constant')
			
			utf_all.append(utf)
	utf_all=np.asarray(utf_all)
	return utf_all

			


if __name__ == '__main__':
	y, bangla_text = read_file(DATA_PATH)
	utf_all = char_to_utf(bangla_text, 100)
	#print(y)
	#print(bangla_text)

	print(utf_all)
