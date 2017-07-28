# coding: utf-8
"""Sequence to Sequence with Keras 1.0"""

from keras.models import Sequential
from keras.layers.core import Dense, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed

def build_model(input_size, max_output_seq_len, hidden_size):
    """建立一个 sequence to sequence 模型"""
    model = Sequential()
    model.add(LSTM(input_dim=input_size, output_dim=hidden_size, return_sequences=False))
    model.add(Dense(hidden_size, activation="relu"))
    # 下面这里将输入序列的向量表示复制 max_output_seq_len 份作为第二个 LSTM 的输入序列
    model.add(RepeatVector(max_output_seq_len))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim=input_size, activation="linear")))
    model.compile(loss="mse", optimizer='adam')

    return model