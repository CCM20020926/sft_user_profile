# 简单用户画像智能体搭建与模型训练实验

**参考论文链接:** http://arxiv.org/abs/2502.10660

**数据集链接:** https://huggingface.co/datasets/Nusrat1234/UserProfileConstruction

## 一、用户画像智能体

**实现代码文件:** [user_profile_agent.py](user_profile_agent.py)

**实现框架：** langchain
## 二、用户画像模型训练

**实现代码文件:** [user_profile_agent.py](train.py)

**选用的基座模型：** Qwen2.5-0.5B-Instruct

**显卡规格:** 单张 40G NVIDIA A100 显卡

**实现框架：** huggingface transformers 库 + trl 库

**查看训练日志命令：**  tensorboard --logdir ./logs  （在项目根目录中执行） 

**需要了解的相关理论：** 监督式指令微调(SFT)、模型训练中常涉及超参数、模型调参算法（其中代表性算法： Adam, AdamW）、学习率调度算法（代表性算法： 余弦退火算法）

### 局限性

1. 当前只使用 loss、token accuracy 指标来衡量模型输出与预期输出接近程度，但未进行语义相似性评估（可使用句子 embbeding模型测量）以及更精细的画像信息字段准确率、精确率、召回率评估

2. 当前训练策略下模型可能存在过拟合问题 （在 eval dataset 上尽管 mean_token_accuracy 逐渐上升， 但loss 却也逐渐上升，不符合常规情形）

### 额外说明

代码总共训练10个回合， 每回合保存一个 checkpoint，但由于采取的是全量微调，每个checkpint会占用 5.6 GB 磁盘空间。可根据自身磁盘空间情况设置 save_total_limit 参数限制 checkpoint 保存数
 
