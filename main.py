import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#read the csv file
url = 'data/stock.csv'
df = pd.read_csv(url)


#Creating a dataframe using Close column
data=df.filter(['Close'])


#coverting dataframe into numpy array
dataset=data.values


#allocating data for training the data
train_data_len=math.ceil(len(dataset)*.8)


#scaling the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)


# creating traning dataset
train_data = scaled_data[0:train_data_len, :]


# splitting into x_train and y_train
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 60:
        print(x_train)
        print(y_train)


#converting x_train and y_train into numpy arrays
x_train,y_train=np.array(x_train),np.array(y_train)


#reshape the data
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


#build the lstm
model= Sequential()
model.add(LSTM(50,return_sequences=True , input_shape= (x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


#compile the model
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=1,epochs=1)


test_data=scaled_data[train_data_len-60: ,:]
#create test datasets
x_test=[]
y_test=dataset[train_data_len:,:]
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])


#convert the data into numpy array
x_test=np.array(x_test)
#reshape the data
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


#predict the price values
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)


#root mean squared error(rmse)
rmse=np.sqrt(np.mean(predictions-y_test)**2)
print(rmse)
