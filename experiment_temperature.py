# -*- coding: utf-8 -*-
# 拓展一：同一开头在不同 temperature 下生成，比较"稳定"与"有创意"的差别
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽TF找不到GPU库的日志
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from data_process import tokenizer
from utils import generate_poetry

MODEL_PATH = "tang_poetry_model.h5"
OPENINGS = ["春", "月", "夜"]
TEMPERATURES = [0.5, 1.0, 1.5]
N_PER_GROUP = 2
TOTAL_COUNT = 20

model = tf.keras.models.load_model(MODEL_PATH)
print(f"[加载] 已加载模型 {MODEL_PATH}")

rows = []
for t in TEMPERATURES:
    for s in OPENINGS:
        for k in range(1, N_PER_GROUP + 1):
            poem, prob = generate_poetry(tokenizer, model, s,
                                         total_count=TOTAL_COUNT,
                                         temperature=t,
                                         return_prob=True)
            print(f"[生成] temperature={t} 开头「{s}」第 {k} 段：{poem}")
            rows.append({"temperature": t, "开头": s,
                         "第几段": k, "生成结果": poem,
                         "平均字概率": f"{prob:.4f}"})

os.makedirs("results", exist_ok=True)
with open("results/temperature_comparison.csv", "w", encoding="utf-8-sig", newline="") as f:
    f.write("temperature,开头,第几段,生成结果,平均字概率\n")
    for r in rows:
        f.write(f"{r['temperature']},{r['开头']},{r['第几段']},"
                f"{r['生成结果']},{r['平均字概率']}\n")

with open("results/temperature_comparison.md", "w", encoding="utf-8") as f:
    f.write("| temperature | 开头 | 生成结果 | 平均字概率 |\n|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['temperature']} | {r['开头']} | {r['生成结果']} "
                f"| {r['平均字概率']} |\n")

# 各 temperature 组的平均字概率汇总
print("\n[汇总] 各 temperature 的平均字概率：")
for t in TEMPERATURES:
    ps = [float(r["平均字概率"]) for r in rows if r["temperature"] == t]
    print(f"  temperature={t}: {sum(ps) / len(ps):.4f}")
print("[完成] 结果见 results/temperature_comparison.csv 和 results/temperature_comparison.md")
