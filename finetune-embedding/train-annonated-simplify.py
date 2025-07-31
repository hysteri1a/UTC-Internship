import logging
import json
import traceback
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample, util
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from data_generation1 import load_daily_dataset, load_data

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

batch_size = 2
lr = 5e-6
max_epochs = 3
eval_every_steps = 1000
early_stop_thresh = 0.50
model_name = "bge-m3"
stage1_dataset = 'data/answer2problem1.json'
stage2_dataset = 'data/annonated_dataset.json'
num_workers = 8
partial_stage1_save = "partial_stage1_dataset.json"
modified_neg_path = "../data/modified_daily_datasets.json"

problems_db, clean_inputs = load_data()
daily_neg_queries = load_daily_dataset()

with open(modified_neg_path, 'r', encoding='utf-8') as f:
    modified_neg_data = json.load(f)
modified_neg_inputs = [item["conversation"][0]["input"] for item in modified_neg_data]

model = SentenceTransformer(model_name, device=device)
for param in model._first_module().auto_model.embeddings.parameters():
    param.requires_grad = False
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

class NestedContrastiveWithNoisePenalty(nn.Module):
    def __init__(self, model, lambda_noise=0.5, margin=0.3):
        super().__init__()
        self.contrastive = losses.OnlineContrastiveLoss(
            model=model,
            distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
            margin=margin
        )
        self.lambda_noise = lambda_noise

    def forward(self, sentence_features, labels):
        c_loss = self.contrastive(sentence_features, labels)
        noise_loss = torch.tensor(0.0, device=c_loss.device)
        return c_loss + self.lambda_noise * noise_loss, c_loss.detach(), noise_loss.detach()

nested_loss = NestedContrastiveWithNoisePenalty(model).to(device)

def eval_noise_similarity(model, noise_queries, problems_db, device):
    model.eval()
    with torch.no_grad():
        q_embs = model.encode(noise_queries, batch_size=batch_size, convert_to_tensor=True).to(device)
        p_embs = model.encode([i['problem'] for i in problems_db], batch_size=batch_size, convert_to_tensor=True).to(device)
        # sim_mat 形状: [num_queries, num_problems]
        sim_mat = util.cos_sim(q_embs, p_embs)
        # 直接对矩阵所有元素取均值
        return sim_mat.mean().item()

# DataLoader
def make_dataloader(examples):
    return DataLoader(
        examples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=model.smart_batching_collate,
        pin_memory=False,
        persistent_workers=False,
        num_workers=0
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

losses_list = []
steps_list = []
best_noise_sim = float('inf')

# Stage1
logging.info("Loading Stage1 dataset...")
with open(stage1_dataset, 'r', encoding='utf-8') as f:
    triples = json.load(f)

# 分离正负样本
positive_samples = [(a, b, l) for a, b, l in triples if l == 1.0]
negative_samples = [(a, b, l) for a, b, l in triples if l == 0.0]
# 保证每 500 条正样本对应一条负样本
max_total = len(positive_samples) * 1000
neg_limit = min(len(negative_samples), max_total - len(positive_samples))
selected_negatives = random.sample(negative_samples, neg_limit)
# 合并并打乱
balanced_triples = positive_samples + selected_negatives
random.shuffle(balanced_triples)
# 限制总样本数：<=210000，且为500的整数倍，且至少保留一条正样本
# 计算可用大小
max_size = min(len(balanced_triples) - len(balanced_triples) % 1000, 210000)
# 保证有一条正样本
if max_size > 0 and positive_samples:
    guaranteed_pos = random.choice(positive_samples)
    pool = [t for t in balanced_triples if t != guaranteed_pos]
    sampled = random.sample(pool, max_size - 1)
    sampled.append(guaranteed_pos)
else:
    sampled = balanced_triples[:max_size]
random.shuffle(sampled)
# 转换为 InputExample
train_examples = [InputExample(texts=[a, b], label=l) for a, b, l in sampled]
total_examples = len(train_examples)
current_stage1 = train_examples.copy()
train_loader = make_dataloader(current_stage1)

logging.info("Starting Stage1 training...")
try:
    model.train()
    step = 0
    for epoch in range(1, max_epochs+1):
        train_loader = make_dataloader(current_stage1)
        for sentence_features, labels in train_loader:
            step += 1
            sentence_features = [{k: v.to(device) for k, v in feat.items()} for feat in sentence_features]
            labels = labels.to(device)
            loss, c_loss, n_loss = nested_loss(sentence_features, labels)

            losses_list.append(loss.item())
            steps_list.append(step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % eval_every_steps == 0:
                noise_sim = eval_noise_similarity(model, daily_neg_queries, problems_db, device)

                if step > 30000 and noise_sim < best_noise_sim:
                    best_noise_sim = noise_sim
                    save_path = f"best_model_step{step}_noise{noise_sim:.4f}"
                    os.makedirs(save_path, exist_ok=True)
                    model.save(os.path.abspath(save_path))
                    logging.info(f"Saved best model to {save_path} with noise_sim={noise_sim:.4f}")

                logging.info(f"[Stage1 Step {step}/{total_examples}] noise_sim={noise_sim:.4f}")
                if noise_sim > early_stop_thresh:
                    # 构建一轮负样本训练集
                    noise_examples = mine_hard_negatives_for_batch(
                        model, sentence_features, modified_neg_inputs, k=1500
                    )

                    noise_loader = make_dataloader(noise_examples)
                    model.train()
                    for noise_sf, noise_lbl in noise_loader:
                        noise_sf = [{k: v.to(device) for k, v in feat.items()} for feat in noise_sf]
                        noise_lbl = noise_lbl.to(device)
                        noise_loss, _, _ = nested_loss(noise_sf, noise_lbl)
                        optimizer.zero_grad()
                        noise_loss.backward()
                        optimizer.step()

                    logging.info(f"Trained on {len(noise_examples)} negative samples due to high noise_sim.")
except Exception:
    logging.error("Exception during Stage1, saving partial dataset...")
    traceback.print_exc()
    raw = [(ex.texts[0], ex.texts[1], ex.label) for ex in current_stage1]
    with open(partial_stage1_save, 'w', encoding='utf-8') as fo:
        json.dump(raw, fo, ensure_ascii=False, indent=2)
    logging.info(f"Partial Stage1 dataset saved to {partial_stage1_save}")
    raise
else:
    logging.info("Stage1 completed successfully.")

# Stage2
logging.info("Loading Stage2 dataset...")
with open(stage2_dataset, 'r', encoding='utf-8') as f:
    triples2 = json.load(f)
# 分离正负样本
positive_samples = [(a, b, label) for a, b, label in triples if label == 1.0]
negative_samples = [(a, b, label) for a, b, label in triples if label == 0.0]

# 确保每 500 条样本有一条正样本
max_total = len(positive_samples) * 1000
max_negatives = min(len(negative_samples), max_total - len(positive_samples))

# 截取负样本
selected_negatives = random.sample(negative_samples, max_negatives)

# 计算可用大小
max_size = min(len(balanced_triples) - len(balanced_triples) % 1000, 210000)
# 保证有一条正样本
if max_size > 0 and positive_samples:
    guaranteed_pos = random.choice(positive_samples)
    pool = [t for t in balanced_triples if t != guaranteed_pos]
    sampled = random.sample(pool, max_size - 1)
    sampled.append(guaranteed_pos)
else:
    sampled = balanced_triples[:max_size]
random.shuffle(sampled)
stage2_base = [InputExample(texts=[a, b], label=label) for a, b, label in sampled]
current_stage2 = stage2_base.copy()
total_examples = len(stage2_base)
train_loader = make_dataloader(current_stage2)

logging.info("Starting Stage2 training...")
model.train()
global_step = 0
for epoch in range(1, max_epochs+1):
    train_loader = make_dataloader(current_stage2)
    for step, (sentence_features, labels) in enumerate(train_loader, start=1):
        global_step += 1
        sentence_features = [{k: v.to(device) for k, v in feat.items()} for feat in sentence_features]
        labels = labels.to(device)
        loss, c_loss, n_loss = nested_loss(sentence_features, labels)

        losses_list.append(loss.item())
        steps_list.append(global_step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_step % eval_every_steps == 0:
            noise_sim = eval_noise_similarity(model, daily_neg_queries, problems_db, device)
            logging.info(f"[Stage2 Step {global_step}/{total_examples}] noise_sim={noise_sim:.4f}")
            if step > 50000 and noise_sim < best_noise_sim:
                best_noise_sim = noise_sim
                save_path = f"best_model2_step{step}_noise{noise_sim:.4f}"
                os.makedirs(save_path, exist_ok=True)
                model.save(os.path.abspath(save_path))
                logging.info(f"Saved best model to {save_path} with noise_sim={noise_sim:.4f}")
            if noise_sim > early_stop_thresh:
                # 构建一轮负样本训练集
                noise_examples = mine_hard_negatives_for_batch(
                    model, sentence_features, modified_neg_inputs, k=1000
                )
                noise_loader = make_dataloader(noise_examples)
                model.train()
                for noise_sf, noise_lbl in noise_loader:
                    noise_sf = [{k: v.to(device) for k, v in feat.items()} for feat in noise_sf]
                    noise_lbl = noise_lbl.to(device)
                    noise_loss, _, _ = nested_loss(noise_sf, noise_lbl)
                    optimizer.zero_grad()
                    noise_loss.backward()
                    optimizer.step()

                logging.info(f"Trained on {len(noise_examples)} negative samples due to high noise_sim.")
    logging.info(f"Stage2 Epoch {epoch} completed.")

out_dir = "finetuned_model_final"
os.makedirs(out_dir, exist_ok=True)
model.save(os.path.abspath(out_dir))
logging.info(f"Training done. Model saved to {out_dir}")

# 最后，在训练结束后画图：
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(steps_list, losses_list, label='Total Loss', color='blue')
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Loss Curve Over Training")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve_train_annonated.png")
plt.show()
