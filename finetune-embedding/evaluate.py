import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

# ========= 1. 配置 Matplotlib =========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False   # 支持负号

DEVICE = 'cuda:2'
torch.cuda.set_device(2)

# ========= 2. 加载问题库 =========
def load_data(json_path: str):
    """
    从 JSON 文件加载问题列表。假设文件格式为：
    [
      {"problems": "问1", …},
      {"problems": "问2", …},
      ...
    ]
    返回所有问题的列表。
    """
    with open(json_path, encoding="utf-8") as f:
        items = json.load(f)

    problems = []
    for entry in items:
        prob = entry.get("problems")
        if prob is None:
            continue
        # 如果 problems 字段已经是列表，可做兼容处理
        if isinstance(prob, list):
            problems.extend(prob)
        else:
            problems.append(prob)
    return problems

# ========= 3. 初始化模型 =========
base_model      = SentenceTransformer(r'../similarity_model',device = DEVICE)
finetuned_model = SentenceTransformer(r'model2', device = DEVICE)

# ========= 4. 测试用例 =========
test_queries = [
    "已经安装了2013完整版的office，客户端检测为什么还是不通过",
    "开通了云盾业务，为什么签署失败",
    "如何对电子钥匙中的信息进行变更",
    "[图片]。 您好，我们在E链网 国网福建上签章，用的是ECP2.0的签章，密码也是。但是一直显示PIN码错误。麻烦看下。U盾的客户端吗。[图片]。 这个是不是一样的。 我没下载客户端。",
    "您好，我想问下，优泰电子钥匙续期的话，有什么流程？"
]
noise_queries = [
    "今天天气怎么样？",
    "如何烹饪意大利面？",
    "世界杯冠军是哪一年？",
    "你是什么东西",
    "最近上映了哪些电影？"
]
test_queries = test_queries + noise_queries  # 合并正负样本

# ========= 5. 相似度计算并返回 Top1 分数与索引 =========
def get_top_scores_and_indices(model: SentenceTransformer,
                               queries: list[str],
                               problems: list[str]
                              ) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 (scores, indices)，
    scores: shape (len(queries),) 的 Top1 相似度分数数组
    indices: shape (len(queries),) 的 Top1 问题在 problems 列表中的下标
    """
    q_embs = model.encode(queries, convert_to_tensor=True)
    p_embs = model.encode(problems, convert_to_tensor=True)
    sim_matrix = util.cos_sim(q_embs, p_embs)
    max_vals, max_idxs = torch.max(sim_matrix, dim=1)
    return max_vals.cpu().numpy(), max_idxs.cpu().numpy()

# ========= 6. 主流程 =========
if __name__ == "__main__":
    problems = load_data("../data/data.json")

    # 6.2 计算 Top1 分数和索引
    base_scores,  base_idxs  = get_top_scores_and_indices(base_model,      test_queries, problems)
    tuned_scores, tuned_idxs = get_top_scores_and_indices(finetuned_model, test_queries, problems)

    # 6.3 打印每条 query 对应的最相似问题
    print("=== 基础模型 Top1 匹配 ===")
    for q, score, idx in zip(test_queries, base_scores, base_idxs):
        print(f"> 查询: {q}\n  最相似: {problems[idx]}  (score={score:.4f})\n")

    print("=== 微调模型 Top1 匹配 ===")
    for q, score, idx in zip(test_queries, tuned_scores, tuned_idxs):
        print(f"> 查询: {q}\n  最相似: {problems[idx]}  (score={score:.4f})\n")

    # 6.4 可视化对比（可选）
    x = np.arange(len(test_queries))
    width = 0.35

    plt.figure(figsize=(14, 8))
    rects1 = plt.bar(x - width/2, base_scores,  width, label='基础模型',  edgecolor='black')
    rects2 = plt.bar(x + width/2, tuned_scores, width, label='微调模型', edgecolor='black')

    plt.title('模型微调前后相似度对比', fontsize=16, pad=20)
    plt.xlabel('测试用例', fontsize=12)
    plt.ylabel('余弦相似度', fontsize=12)
    plt.xticks(x, [f"用例{i+1}" for i in range(len(test_queries))], rotation=25)
    plt.ylim(0, 1.15)

    for rect in list(rects1) + list(rects2):
        h = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2, h, f'{h:.2f}',
                 ha='center', va='bottom', fontsize=10)

    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    out_path = Path("model_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至 {out_path.resolve()}")
    plt.show()
