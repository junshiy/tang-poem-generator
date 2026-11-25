# -*- coding: utf-8 -*-
# 拓展二：指定关键词生成藏头诗
# 用法1：python generate_keyword.py 明月   （命令行传入关键词，生成一首并退出）
# 用法2：python generate_keyword.py        （先生成批量结果表，再进入手动输入模式）
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽TF找不到GPU库的日志
import warnings
warnings.filterwarnings('ignore')
import sys
import tensorflow as tf
from data_process import tokenizer
from utils import generate_keyword_poetry

MODEL_PATH = "tang_poetry_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)
print(f"[加载] 已加载模型 {MODEL_PATH}")

if len(sys.argv) > 1:
    # 命令行模式
    keyword = sys.argv[1]
    poem = generate_keyword_poetry(tokenizer, model, keyword,
                                   temperature=1.0)
    print(f"[生成] 关键词「{keyword}」：{poem}")
    sys.exit(0)

# 批量模式：生成结果表
KEYWORDS = ["明月", "春风", "故乡", "山河", "秋水"]
rows = []
for kw in KEYWORDS:
    poem = generate_keyword_poetry(tokenizer, model, kw,
                                   temperature=1.0)
    print(f"[生成] 关键词「{kw}」：{poem}")
    rows.append({"关键词": kw, "生成结果": poem})

os.makedirs("results", exist_ok=True)
with open("results/keyword_poems.csv", "w", encoding="utf-8-sig",
          newline="") as f:
    f.write("关键词,生成结果\n")
    for r in rows:
        f.write(f"{r['关键词']},{r['生成结果']}\n")

with open("results/keyword_poems.md", "w", encoding="utf-8") as f:
    f.write("| 关键词 | 生成结果 |\n|---|---|\n")
    for r in rows:
        f.write(f"| {r['关键词']} | {r['生成结果']} |\n")

print(f"[完成] 已生成 {len(rows)} 条，见 results/keyword_poems.csv 和 .md")

# 手动输入模式：现场演示用，输入关键词即生成，直接回车退出
print("\n===== 手动输入模式 =====")
while True:
    keyword = input("请输入关键词（1~4 个字，直接回车退出）：").strip()
    if not keyword:
        print("已退出。")
        break
    try:
        poem = generate_keyword_poetry(tokenizer, model, keyword,
                                       temperature=1.0)
        print(f"[生成] 关键词「{keyword}」：{poem}\n")
    except ValueError as e:
        print(f"[失败] {e}\n")
