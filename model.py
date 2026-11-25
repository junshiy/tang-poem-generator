# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import TimeDistributed
from data_process import tokenizer


def get_model():
    model = Sequential([
        Input((None,)),
        Embedding(input_dim=tokenizer.vocab_size,
                  output_dim=64),
        LSTM(128, return_sequences=True),
        # 每个时间步都输出词表上的概率分布
        TimeDistributed(
            Dense(tokenizer.vocab_size,
                  activation='softmax')
        )
    ])
    return model
