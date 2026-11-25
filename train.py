# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽TF找不到GPU库的日志
import warnings
warnings.filterwarnings('ignore')
from data_process import tokenizer, get_dataset
from model import get_model
from utils import generate_poetry
import tensorflow as tf

EPOCHS = 50
BATCH_SIZE = 128

x_train, y_train = get_dataset()
print(f"[数据] 训练样本 x: {x_train.shape}, y: {y_train.shape}")

model = get_model()
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.sparse_categorical_crossentropy)


class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lowest = 1e10
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        self.history.append(loss)
        # 保存 loss 最低的模型
        if loss <= self.lowest:
            self.lowest = loss
            model.save("tang_poetry_model.h5")
            print(f"[保存] epoch {epoch + 1} loss={loss:.4f}")
        print(f"[epoch {epoch + 1}] loss={loss:.4f}")
        print(generate_poetry(tokenizer, model))


history_cb = Evaluate()
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=[history_cb])

# 导出每轮 loss，供画曲线用
with open("results/loss_record.txt", "w", encoding="utf-8") as f:
    for i, loss in enumerate(history_cb.history, 1):
        f.write(f"{i}\t{loss:.6f}\n")
print("[完成] loss 记录已写入 results/loss_record.txt")
