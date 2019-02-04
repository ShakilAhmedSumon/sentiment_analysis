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


if __name__ == '__main__':
	y, bangla_text = read_file(DATA_PATH)
	print(y)
	print(bangla_text)
