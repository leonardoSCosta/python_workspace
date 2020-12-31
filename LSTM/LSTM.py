#!/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import plot_model

training_data = pd.read_csv('./FB_Train.csv')
training_data = training_data.iloc[:, 4].values


scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data.reshape(-1, 1))
# print(training_data)

x_training_data = []
y_training_data = []

# Dados dos últimos quarenta dias e do dia atual (x,y)
for i in range(40, len(training_data)):
    x_training_data.append(training_data[i-40:i, 0])
    y_training_data.append(training_data[i, 0])

x_training_data = np.array(x_training_data)
y_training_data = np.array(y_training_data)

# Reshape por causa do tensorflow
x_training_data = np.reshape(x_training_data, (x_training_data.shape[0],
                                               x_training_data.shape[1], 1))

print(x_training_data.shape, y_training_data.shape)

rnn = Sequential()

# Adicionando a primada camada LSTM
rnn.add(LSTM(units=45, return_sequences=True,
             input_shape=(x_training_data.shape[1], 1)))

# Adicionando um dropout de 20% para evitar o over fitting
rnn.add(Dropout(0.2))


# Adicionando as outras camadas

for i in [True, True, False]:
    rnn.add(LSTM(units=45, return_sequences=i))
    rnn.add(Dropout(0.2))

# Adicionando a camada de saída
rnn.add(Dense(units=1))

plot_model(rnn, to_file='./rnn.png', show_shapes=True)

# Compilando a rede
rnn.compile(optimizer='adam', loss='mean_squared_error')

# Treinando a rede
rnn.fit(x_training_data, y_training_data, epochs=250, batch_size=32)

# Fazendo as predições
test_data = pd.read_csv('./FB_Test.csv')
test_data = test_data.iloc[:, 4].values

# plt.plot(test_data)

# Dados sem transformação
unscaled_training_data = pd.read_csv('./FB_Train.csv')
unscaled_test_data = pd.read_csv('./FB_Test.csv')

# Juntando os dados dos para realizar a predição, já que são necessários os dados
# dos últimos 40 dias
all_data = pd.concat(
    (unscaled_training_data['Close'], unscaled_test_data['Close']), axis=0)

x_test_data = all_data[len(all_data) - len(test_data) - 40:].values

x_test_data = np.reshape(x_test_data, (-1, 1))
x_test_data = scaler.transform(x_test_data)

final_x_test_data = []

for i in range(40, len(x_test_data)):
    final_x_test_data.append(x_test_data[i-40:i, 0])
final_x_test_data = np.array(final_x_test_data)

# Reshape por causa do tensorflow
final_x_test_data = np.reshape(final_x_test_data, (final_x_test_data.shape[0],
                                                   final_x_test_data.shape[1], 1))

# Predições
predictions = rnn.predict(final_x_test_data)
unscaled_predictions = scaler.inverse_transform(predictions)

# print('Test data len', len(x_test_data))
plt.plot(unscaled_predictions, color='#135485', label='Predictions')
plt.plot(test_data, color='black', label="Real Data")
plt.title("Facebook Stock Price Predictions")
plt.show()
