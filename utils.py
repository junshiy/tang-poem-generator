# -*- coding: utf-8 -*-
import numpy as np


# temperature 越小采样越保守，越大越随机
def random_max(preds, temperature=1.0):
    preds = np.array(preds).astype('float64')
    preds[0] = 0  # 屏蔽 <UNK>（词表外生僻字），不允许被采样
    preds = preds / np.sum(preds)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# 以 s 开头续写，总字数 total_count（默认20字=两句五言）
# return_prob=True 时返回 (文本, 生成字的平均概率)
def generate_poetry(tokenizer, model, s=None,
                    total_count=20, temperature=1.5,
                    return_prob=False):
    if s is None or s == '':
        s = tokenizer.random_a_char()
    id_seq = tokenizer.encode(s)
    count = len(s)
    probs = []
    for i in range(count, total_count):
        pred = model.predict(np.array([id_seq]), verbose=0)[0, -1]
        pred_index = random_max(pred, temperature)
        probs.append(float(pred[pred_index]))
        id_seq.append(pred_index)
        s = s + tokenizer.id_to_char(pred_index)
        count += 1
        if count % 10 == 0:
            s = s + '。'
        elif count % 5 == 0:
            s = s + '，'
    if return_prob:
        return s, float(np.mean(probs))
    return s


# 指定关键词生成：关键词每个字固定在各句（5字）开头，其余位置正常采样
def generate_keyword_poetry(tokenizer, model, keyword,
                            temperature=1.0, total_count=20):
    heads = list(keyword)
    if len(heads) > 4:
        print(f"[提示] 关键词超过4个字，只使用前4个：{''.join(heads[:4])}")
        heads = heads[:4]
    for ch in heads:
        if tokenizer.char_to_id(ch) == 0:
            raise ValueError(f"字「{ch}」不在词表中，请换一个关键词")
    id_seq = []
    s = ""
    head_pos = 0
    count = 0
    while count < total_count:
        if head_pos < len(heads) and count % 5 == 0:
            # 句首固定用关键词的字
            ch = heads[head_pos]
            head_pos += 1
            pred_index = tokenizer.char_to_id(ch)
        else:
            pred = model.predict(np.array([id_seq]), verbose=0)[0, -1]
            pred_index = random_max(pred, temperature)
        id_seq.append(pred_index)
        s = s + tokenizer.id_to_char(pred_index)
        count += 1
        if count % 10 == 0:
            s = s + '。'
        elif count % 5 == 0:
            s = s + '，'
    return s


# 藏头诗：head 中每个字各起一句五言
def generate_acrostic(tokenizer, model, head):
    poem = ""
    for s in head:
        poem = poem + generate_poetry(tokenizer,
                                      model,
                                      s,
                                      total_count=5)
    return poem
