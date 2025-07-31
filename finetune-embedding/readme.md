# 微调embedding模型 使用说明

该 项目 提供了一套完整的针对bge-m3的微调流程代码。

> ⚙️ **环境准备**：
>
> 1. 创建独立 Python 环境：`python -m venv .venv` 或 `conda create -n py310 python=3.10`
> 2. 安装依赖：
>    ```bash
>    pip install -r requirements.txt
>    ```
> 3. 代码位置：`/home/user/Script/test/finetune`


项目目录结构
  ├── origin_data/                          # 原数据集
  ├── data/                                 # 处理后数据集
  │   ├── parsed_data.json                  # 标准问答
  │   ├── output.json/                      # 真实对话（无标签）
  │   ├── modified_daily_datasets.json      # 处理后日常对话，用于降低无关内容相似度、测试微调后模型、增强泛化能力
  │   ├── annonated_dataset.json            # 人工标注数据集
  │   ├── answer2problem1.json              # 训练用数据集（生成的标签）
  │   └── data.json                         # 很原始的数据
  ├── finetuned_model_nested_loss           # 一个微调后的embedding模型
  ├── Qwen2.5-VL-32B-Instruct-AWQ           # qwen32B模型
  ├── base2.xlsx                            # 参考回答
  ├── similarity_model                      # bge-m3嵌入模型
  ├── train-annonated-simplify.py           # 带人工标注的微调
  └── train1-simplify.py                    # 不带人工标注的微调

---


## 1. preprocess.py

预处理 base2.xlsx 并保存为json文件 ./data/parsed_data.json

---

## 2. compare.py

比较微调前和微调后的模型的层数，标记参数改变了的层数。

注意文件夹名称。

---

## 3. evaluate.py

比较微调模型与原始模型在 in-distribution(ID)和out-of-distribution(OOD)上的表现差异

我们希望对于无关的，相似度越低越好；对于相关的，相似度越高越好。

---

## 4. expert-annonated.py

手动标注数据集, 标注完后进行微调，然后保存模型。

---

## 5. train1-simplify.py

单阶段训练脚本，主要功能：

模型训练：使用BGE-M3模型进行对比学习训练
数据处理：加载answer2problem1.json数据集，平衡正负样本比例（正样本:负样本 = 1000:1）
动态负样本挖掘：当噪声相似度超过阈值(0.5)时，动态挖掘硬负样本进行额外训练
评估机制：每1000步评估一次噪声相似度，保存最佳模型
损失函数：使用自定义的NestedContrastiveLoss（基于OnlineContrastiveLoss）
---

## 6. train-annonated-simplify.py

两阶段训练脚本，主要功能：

Stage1训练：使用answer2problem1.json数据集进行初始训练
Stage2训练：使用annonated_dataset.json数据集进行精细化训练
嵌入层冻结：在第二段代码中冻结了embedding层参数
增强损失函数：使用NestedContrastiveWithNoisePenalty，包含噪声惩罚项
分阶段策略：两个阶段使用不同的数据集和训练策略

主要区别

训练阶段：第一段是单阶段，第二段是两阶段训练
数据集：第二段使用了额外的注释数据集
模型参数：第二段冻结了embedding层
损失函数：第二段添加了噪声惩罚机制

---

## 7. 数据文件生成

data_generation1.py : annonated_dataset.json  # 生产了标注数据集
data_generation2.py : answer2problem.json   # 负例少一点
data_generation3.py : answer2problem1.json  # 负例多一点

---
