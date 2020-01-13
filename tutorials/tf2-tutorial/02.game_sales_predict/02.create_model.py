import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

training_data_df = pd.read_csv("dataset/sales_data_training_scaled.csv")

X = training_data_df.drop('销售总额', axis=1).values
Y = training_data_df[['销售总额']].values

# 定义模型
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss="mean_squared_error", optimizer="adam")
