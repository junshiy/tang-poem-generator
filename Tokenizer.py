# -*- coding: utf-8 -*-
import numpy as np


class Tokenizer:
    def __init__(self, chars):
        self.index_char = ['<UNK>'] + chars  # 0号留给词表外的字
        self.char_index = dict(
            (c, i) for i, c in enumerate(self.index_char)
        )
        self.vocab_size = len(self.index_char)

    def char_to_id(self, char):
        return self.char_index.get(
            char,
            self.char_index['<UNK>'])

    def id_to_char(self, index):
        return self.index_char[index]

    def random_a_char(self):
        return np.random.choice(self.index_char[1:])

    def encode(self, text):
        return [self.char_to_id(c) for c in text]
