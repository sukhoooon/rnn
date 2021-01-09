import numpy as np
from sklearn.metrics import accuracy_score
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import LSTM

# parameters for data load
num_words = 300
maxlen = 50
test_split = 0.3
(X_train_1, y_train), (X_test_1, y_test) = reuters.load_data(num_words = num_words, maxlen = maxlen, test_split = test_split)

# pad the sequences with zeros
# padding parameter is set to 'post' => 0's are appended to end of sequences
X_train = pad_sequences(X_train_1, padding = 'post')
X_test = pad_sequences(X_test_1, padding = 'post')

X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# #  여기는 자연어 특성 때문에 하는거임
# 합치기
y_data = np.concatenate((y_train, y_test))
# one-hot encoding
y_data = to_categorical(y_data)

y_train = y_data[:X_train.shape[0]]
y_test = y_data[X_train.shape[0]:]
# inverse one-hot encoding
y_test_ = np.argmax(y_test, axis = 1)

# (1395, 49, 1)
# (599, 49, 1)

# 1. vanilla
# def vanilla_rnn():
#     model = Sequential()
#     model.add(SimpleRNN(50, input_shape=(49, 1), return_sequences=False))
#     model.add(Dense(46))
#     model.add(Activation('softmax'))
#
#     adam = optimizers.Adam(lr=0.001)
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#
#     return model
# model = KerasClassifier(build_fn = vanilla_rnn, epochs = 200, batch_size = 50, verbose = 1)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# y_test_ = np.argmax(y_test, axis = 1)
# print(accuracy_score(y_pred, y_test_))

# 2. stacked vanilla
# def stacked_vanilla_rnn():
#     model = Sequential()
#     model.add(SimpleRNN(50, input_shape=(49, 1),
#                         return_sequences=True))  # return_sequences parameter has to be set True to stack
#     model.add(SimpleRNN(50, return_sequences=False))
#     model.add(Dense(46))
#     model.add(Activation('softmax'))
#
#     adam = optimizers.Adam(lr=0.001)
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#
#     return model
# model = KerasClassifier(build_fn = stacked_vanilla_rnn, epochs = 200, batch_size = 50, verbose = 1)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(accuracy_score(y_pred, y_test_))

# 3. lstm
# def lstm():
#     model = Sequential()
#     model.add(LSTM(50, input_shape=(49, 1), return_sequences=False))
#     model.add(Dense(46))
#     model.add(Activation('softmax'))
#
#     adam = optimizers.Adam(lr=0.001)
#     model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#
#     return model
# model = KerasClassifier(build_fn = lstm, epochs = 200, batch_size = 50, verbose = 1)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(accuracy_score(y_pred, y_test_))

# 4. stacked lstm
def stacked_lstm():
    model = Sequential()
    model.add(LSTM(50, input_shape=(49, 1), return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(46))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model
model = KerasClassifier(build_fn = stacked_lstm, epochs = 200, batch_size = 50, verbose = 1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_pred, y_test_))
