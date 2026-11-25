# AI「小诗人」——唐诗自动生成

《自然语言处理》期末课程设计 · 项目工程规范版

## 简介

基于课程提供的唐诗语料（poetry.txt，约 7.2 万条五言诗句），训练字符级 LSTM 语言模型：给它一个开头，逐字预测下一个字，自动续写五言诗；另实现指定关键词的藏头诗生成、temperature 解码策略对比实验。

## 文件结构

```
AI小诗人/
├── data_process.py          # 预处理：去标点→保留10字句→过滤低频字→构造序列
├── Tokenizer.py             # 字符级分词器（字符↔编号）
├── model.py                 # 模型：Embedding(64)→LSTM(128)→TimeDistributed(Dense+softmax)
├── train.py                 # 训练脚本，loss 最低时保存模型
├── utils.py                 # 生成工具：temperature 采样 / 续写 / 关键词藏头
├── generate_table.py        # 基础任务：8 个开头批量生成结果表
├── experiment_temperature.py# 拓展一：temperature 0.5/1.0/1.5 对比
├── generate_keyword.py      # 拓展二：指定关键词藏头诗
├── show_model.py            # 打印模型结构
├── poetry.txt               # 唐诗语料（课程提供）
├── tang_poetry_model.h5     # 训练好的模型（loss 最低 4.7448）
├── results/                 # 全部实验结果
│   ├── generation_results.csv/.md   基础生成结果表（8 开头×2）
│   ├── keyword_poems.csv/.md        关键词藏头诗结果表
│   ├── temperature_comparison.csv/.md  temperature 对比表
│   ├── temperature_analysis.md       temperature 对比分析
│   ├── sample_analysis.md            最好/最差样本分析
│   ├── loss_record.txt / loss_curve.png  训练 loss 记录与曲线
│   └── training_log.txt              完整训练日志
└── model/                   # 从 .h5 导出的纯前端推理权重（供在线体验页）
    ├── vocab.json           词表（3745 字）
    ├── weights_meta.json    各权重张量形状
    └── *.bin                float32 权重：emb/lstm_k/lstm_u/lstm_b/dense_k/dense_b
```

> 命名说明：原项目文件名为中文（如「我的唐诗.h5」「生成结果表.csv」），已统一改为英文规范命名，代码内路径引用同步更新，内容与结果不变。

## 关键参数

| 参数 | 取值 |
|---|---|
| 训练诗句 | 72405 条（去标点后恰好 10 字） |
| 词表 | 3745 字（含 `<UNK>`，过滤词频 < 5） |
| 模型 | Embedding(64) → LSTM(128) → Dense(softmax) |
| 训练 | Adam，batch=128，50 轮，最低 loss 4.7448 |
| temperature | 0.5 / 1.0 / 1.5 对比 |

## 实验结论

- 字符级 LSTM 学会了五言诗的外形（格式 100% 正确、常见搭配合理），但逐字推敲仍有不足——与 50 轮训练量匹配。
- temperature 越小越稳定（0.5 平均字概率 0.1039），越大越有创意但不可读比例上升（1.5 平均字概率 0.0099）。
- 采样阶段屏蔽 0 号 `<UNK>` 并重新归一化，生成结果不再出现词表外标记——无需重训模型。

## 在线体验

`poem-app/index.html` 是纯前端体验页：把上述权重用 JavaScript 手写 LSTM 前向传播在浏览器里真实推理，可输入任意开头、调 temperature 实时生成诗，或生成指定关键词藏头诗，无需后端、无需 TensorFlow。
