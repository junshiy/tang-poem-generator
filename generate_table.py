# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽TF找不到GPU库的日志
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from data_process import tokenizer
from utils import generate_poetry

MODEL_PATH = "tang_poetry_model.h5"
OPENINGS = ["春", "花", "月", "风", "雪", "江", "夜", "人"]
N_PER_OPENING = 2
TOTAL_COUNT = 20

model = tf.keras.models.load_model(MODEL_PATH)
print(f"[加载] 已加载模型 {MODEL_PATH}")

rows = []
for s in OPENINGS:
    for k in range(1, N_PER_OPENING + 1):
        poem = generate_poetry(tokenizer, model, s,
                               total_count=TOTAL_COUNT)
        print(f"[生成] 开头「{s}」第 {k} 段：{poem}")
        rows.append({"开头": s, "第几段": k, "生成结果": poem})

os.makedirs("results", exist_ok=True)
with open("results/generation_results.csv", "w", encoding="utf-8-sig", newline="") as f:
    f.write("开头,第几段,生成结果\n")
    for r in rows:
        f.write(f"{r['开头']},{r['第几段']},{r['生成结果']}\n")

with open("results/generation_results.md", "w", encoding="utf-8") as f:
    f.write("| 开头 | 生成结果 |\n|---|---|\n")
    for r in rows:
        f.write(f"| {r['开头']} | {r['生成结果']} |\n")

print(f"[完成] 已生成 {len(rows)} 条结果，见 results/ 目录")
