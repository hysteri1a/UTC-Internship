import logging
import json
import os
import traceback
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample, util
from torch.cuda.amp import GradScaler, autocast
import random
from data_generation3 import load_daily_dataset, load_data
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# —— 日志配置 ——
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# —— 设备配置 ——
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# —— 超参数 ——
batch_size        = 2
lr                = 5e-6
max_epochs        = 3
eval_every_steps  = 1000
early_stop_thresh = 0.5
model_name        = "bge-m3"
stage1_dataset    = 'data/answer2problem1.json'
num_workers       = 8
partial_save_path = "partial_dataset.json"
modified_neg_path = "../data/modified_daily_datasets.json"

# —— 加载数据 & 负样本查询 ——
problems_db, clean_inputs = load_data()
daily_neg_queries         = load_daily_dataset()
with open(modified_neg_path, 'r', encoding='utf-8') as f:
    modified_neg_data = json.load(f)
daily_neg_inputs = [item["conversation"][0]["input"] for item in modified_neg_data]

# —— 初始化模型 ——
model = SentenceTransformer(model_name, device=device)
# 取消冻结 embedding 以便全局微调
# for param in model._first_module().auto_model.embeddings.parameters():
#     param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = GradScaler()

# —— 自定义损失 ——
class NestedContrastiveLoss(nn.Module):
    def __init__(self, model, margin=0.3):
        super().__init__()
        self.contrastive = losses.OnlineContrastiveLoss(
            model=model,
            distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
            margin=margin
        )

    def forward(self, sentence_features, labels):
        return self.contrastive(sentence_features, labels)

nested_loss = NestedContrastiveLoss(model).to(device)

# DataLoader
def make_dataloader(examples):
    return DataLoader(
        examples,
        shuffle = True,
        batch_size = batch_size,
        collate_fn = model.smart_batching_collate,
        pin_memory = False,
        persistent_workers = False,
        num_workers = 0
    )

# 硬负样本挖矿
def mine_hard_negatives_for_batch(model, batch_sentence_features, neg_corpus, k=5, max_pool_size=10000):
    queries = [
        model.tokenizer.batch_decode(feat['input_ids'], skip_special_tokens=True)[0]
        for feat in batch_sentence_features
    ]
    pool = random.sample(neg_corpus, min(len(neg_corpus), max_pool_size))
    neg_embs = model.encode(pool, batch_size=batch_size, convert_to_tensor=True)

    hard_examples = []
    for query in queries:
        q_emb = model.encode([query], convert_to_tensor=True)
        sims = util.cos_sim(q_emb, neg_embs)[0]
        topk_idx = torch.topk(sims, k=k).indices.tolist()
        for idx in topk_idx:
            hard_examples.append(
                InputExample(texts=[query, pool[idx]], label=0.0)
            )
    return hard_examples

# —— 噪声相似度评估 ——
def eval_noise_similarity(p_embs, q_embs):
    """
    计算所有 query 与所有 problem 的余弦相似度矩阵的全局平均值
    """
    # sim_mat 形状: [num_queries, num_problems]
    sim_mat = util.cos_sim(q_embs, p_embs)
    # 直接对矩阵所有元素取均值
    return sim_mat.mean().item()


losses_list = []
steps_list = []

# —— 加载并平衡训练样本 ——
with open(stage1_dataset, 'r', encoding='utf-8') as f:
    triples = json.load(f)

positive_samples = [(a, b, l) for a, b, l in triples if l == 1.0]
negative_samples = [(a, b, l) for a, b, l in triples if l == 0.0]
# 保证每 500 条正样本对应 1 条负样本
max_total = len(positive_samples) * 1000
neg_limit = min(len(negative_samples), max_total - len(positive_samples))
selected_negatives = random.sample(negative_samples, neg_limit)
balanced_triples = positive_samples + selected_negatives
random.shuffle(balanced_triples)
# 限制总样本数：小于 210k 且为 500 的整数倍，且至少保留 1 个正样本
max_size = min(len(balanced_triples) - len(balanced_triples) % 1000, 262500)
# 随机抽取一条正样本保底
if max_size > 0 and len(positive_samples) > 0:
    guaranteed_pos = random.choice(positive_samples)
    # 抽取其余样本
    pool = [t for t in balanced_triples if t != guaranteed_pos]
    sampled = random.sample(pool, max_size - 1)
    sampled.append(guaranteed_pos)
else:
    sampled = balanced_triples[:max_size]
random.shuffle(sampled)
train_examples = [InputExample(texts=[a, b], label=l) for a, b, l in sampled]

total_examples = len(train_examples)
# —— 训练流程 ——
logging.info("Starting training without Stage1...")

best_noise_sim = float('inf')  # 用于记录最低噪声相似度

try:
    for epoch in range(1, max_epochs + 1):
        step = 0
        train_loader = make_dataloader(train_examples)
        model.train()
        for sentence_features, labels in train_loader:
            step += 1
            sentence_features = [
                {k: v.to(device, non_blocking=True) for k, v in feat.items()}
                for feat in sentence_features
            ]
            labels = labels.to(device, non_blocking=True)

            with autocast():
                loss = nested_loss(sentence_features, labels)

            losses_list.append(loss.item())
            steps_list.append(step)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % eval_every_steps == 0:
                p_embs = model.encode(
                    [item['problem'] for item in problems_db],
                    batch_size=batch_size,
                    convert_to_tensor=True
                ).to(device)
                q_embs_cached = model.encode(
                    daily_neg_queries,
                    batch_size=batch_size,
                    convert_to_tensor=True
                ).to(device)
                noise_sim = eval_noise_similarity(p_embs, q_embs_cached)
                logging.info(f"[Epoch {epoch} Step {step}/{total_examples}] Loss={loss.item():.4f}, noise_sim={noise_sim:.4f}")

                if step > 30000 and noise_sim < best_noise_sim:
                    best_noise_sim = noise_sim
                    save_path = f"best_model3_noise{noise_sim:.4f}"
                    os.makedirs(save_path, exist_ok=True)
                    model.save(os.path.abspath(save_path))
                    logging.info(f"Saved best model to {save_path} with noise_sim={noise_sim:.4f}")

                if noise_sim > early_stop_thresh:
                    logging.info("Noise similarity exceeded threshold, mining hard negatives...")

                    neg_examples = mine_hard_negatives_for_batch(
                        model, sentence_features, daily_neg_inputs, k=1500
                    )
                    neg_loader = make_dataloader(neg_examples)
                    for neg_features, neg_labels in neg_loader:
                        neg_features = [
                            {k: v.to(device, non_blocking=True) for k, v in feat.items()}
                            for feat in neg_features
                        ]
                        neg_labels = neg_labels.to(device, non_blocking=True)

                        with autocast():
                            neg_loss = nested_loss(neg_features, neg_labels)
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(neg_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    logging.info(f"Trained one negative round with {len(neg_examples)} examples")

except Exception:
    logging.error("Exception during training")
    traceback.print_exc()
    raw = [(ex.texts[0], ex.texts[1], ex.label) for ex in train_examples]
    with open(partial_save_path, "w", encoding="utf-8") as fout:
        json.dump(raw, fout, ensure_ascii=False, indent=2)
    logging.info(f"Saved partial dataset ({len(raw)}) to {partial_save_path}")
    raise
else:
    out_dir = "final_model_stage2"
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.abspath(out_dir))
    logging.info(f"Training done. Model saved to {out_dir}")

# 最后，在训练结束后画图：
import matplotlib.pyplot as plt

# 在训练脚本末尾添加以下代码以绘图：
plt.figure(figsize=(10, 6))
plt.plot(steps_list, losses_list, label="Training Loss", color='blue')
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve_train1.png")
plt.show()
