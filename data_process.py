# -*- coding: utf-8 -*-
import re
from collections import Counter
from Tokenizer import Tokenizer
import numpy as np

# 读取诗句，去标点，只保留10个字的诗句（五言两句，定长无需填充）
with open("poetry.txt", encoding='utf-8') as f:
    poetry = f.readlines()
rule = re.compile('[\W]')  # \W 表示非文字
poetry = [rule.sub(repl='', string=x) for x in poetry]
poetry = [x for x in poetry if len(x) == 10]
print(f"[数据] 预处理后保留诗句 {len(poetry)} 条")

# 统计字频，过滤低频字，生成词表
counter = Counter()
for x in poetry:
    counter.update(x)
MIN_WORD_COUNT = 5  # 最小词频，可调
chars = [(char, count) for char, count in counter.items()
         if count >= MIN_WORD_COUNT]
chars = sorted(chars, key=lambda x: -x[1])
chars = [c for c, count in chars]
print(f"[数据] 过滤低频字后词表大小 {len(chars)}")

tokenizer = Tokenizer(chars)


# x 为前9个字，y 为后9个字（错开一位，即"预测下一个字"任务）
def get_dataset():
    x_train = []
    y_train = []
    for p in poetry:
        id_seq = tokenizer.encode(p)
        x_train.append(id_seq[:-1])
        y_train.append(id_seq[1:])
    return np.array(x_train), np.array(y_train)
