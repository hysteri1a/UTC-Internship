import json
import re
import random
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, util, losses, InputExample
import numpy as np
import os

# —— 日志配置 ——
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# —— 手动选择 GPU ——
GPU_ID = 3  # 手动指定 GPU 序号
if torch.cuda.is_available():
    torch.cuda.set_device(GPU_ID)
    device = torch.device(f"cuda:{GPU_ID}")
else:
    device = torch.device("cpu")
logging.info(f"Using device: {device}")

def augment_text(text):
    if len(text) < 2:
        return text
    op = random.choice(['swap', 'delete', 'dup'])
    idx = random.randrange(len(text))
    if op == 'swap' and idx < len(text) - 1:
        lst = list(text)
        lst[idx], lst[idx+1] = lst[idx+1], lst[idx]
        return ''.join(lst)
    elif op == 'delete':
        return text[:idx] + text[idx+1:]
    elif op == 'dup':
        return text[:idx] + text[idx] + text[idx:]
    return text



def load_daily_dataset(path="../data/modified_daily_datasets.json", max_samples=100):
    """
    读取本地日常对话数据集，用于负样本
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    queries = []
    for item in data:
        for msg in item.get('conversation', []):
            inp = msg.get('input')
            if inp:
                queries.append(inp.strip())
    queries = list(set(queries))
    random.shuffle(queries)
    return queries[:max_samples]


def load_data(json_qna_path="../data/parsed_data.json", json_conv_path="../data/output.json"):
    """
    读取 QnA 数据并返回列表，每项为 {'problem','answer'}；
    以及清洗后的对话输入输出列表
    """
    with open(json_qna_path, encoding="utf-8") as f:
        qna = json.load(f)
    problems_db = []
    for item in qna:
        problems_db.append({'problem': item['problems'], 'answer': item.get('answer', '')})

    with open(json_conv_path, encoding="utf-8") as f:
        convs = json.load(f)
    clean_inputs = []
    for conv in convs:
        for msg in conv.get('conversation', []):
            inp = msg.get('input', '').strip()
            out = msg.get('output', '').strip()
            txt = re.sub(r"(客户已接入，会话开始。|会话已超时结束。|\[图片\])", "", inp).strip()
            if txt and out:
                clean_inputs.append((txt, out))
    return problems_db, clean_inputs


def generate_random_negatives(inp_emb, problem_embs, problems_db,
                              num_neg=20, hard_pool=50,
                              neg_queries=None, use_neg=False):
    sims = util.pytorch_cos_sim(inp_emb, problem_embs)[0].cpu().numpy()
    desc_idx = np.argsort(-sims)
    pos_idx = desc_idx[0]
    samples = [(problems_db[pos_idx], 1.0)]
    neg_idxs = random.sample(desc_idx[2: hard_pool+2].tolist(), k=num_neg)
    for idx in neg_idxs:
        samples.append((problems_db[idx], 0.0))
    if use_neg and neg_queries:
        neg_q = random.choice(neg_queries)
        samples.append(({'problem': neg_q, 'answer': ''}, 0.0))
    return samples


def build_dataset_dynamic(model, problems_db, clean_inputs,
                          neg_queries=None,
                          num_neg=100, hard_pool=200,
                          use_neg=False, manual_done=False,
                          ans_threshold=0.79):
    """
    构建训练样本：根据回答与数据库答案相似度决定样本正负
    累计未匹配回答对应的问题前缀，直到相似度超过阈值
    """
    # 预编码答案库
    db_answers = [item['answer'] for item in problems_db]
    ans_embs = model.encode(db_answers, convert_to_tensor=True)
    examples = []
    # 初始化前缀问题
    prefix = ""
    for inp_text, out_text in clean_inputs:
        # 每轮将当前问题累加
        prefix += inp_text
        # 计算回答与答案库相似度
        out_emb = model.encode(out_text, convert_to_tensor=True)
        sims = util.cos_sim(out_emb, ans_embs)[0].cpu().numpy()
        max_idx = int(np.argmax(sims))
        max_sim = sims[max_idx]
        if max_sim < ans_threshold:
            # 若未命中，则生成基于当前 prefix 的随机负样本
            for item, lbl in generate_random_negatives(
                    model.encode(prefix, convert_to_tensor=True),
                    ans_embs, problems_db,
                    num_neg, hard_pool,
                    neg_queries, use_neg):
                examples.append(InputExample(texts=[prefix, item['problem']], label=lbl))
        else:
            # 命中：构建正样本，使用前缀+回答为查询，正例为对应问题
            pos_problem = problems_db[max_idx]['problem']
            query_text = prefix + out_text
            examples.append(InputExample(texts=[query_text, pos_problem], label=1.0))
            # 增加轻微扰动增强
            aug = augment_text(query_text)
            examples.append(InputExample(texts=[aug, pos_problem], label=1.0))
            # 其余负样本
            for item, lbl in generate_random_negatives(
                    model.encode(query_text, convert_to_tensor=True),
                    ans_embs, problems_db,
                    num_neg, hard_pool,
                    neg_queries, use_neg):
                examples.append(InputExample(texts=[query_text, item['problem']], label=lbl))
    return examples



def eval_noise_similarity(model, noise_queries, problems_db, device):
    q_embs = model.encode(noise_queries, convert_to_tensor=True).to(device)
    p_embs = model.encode([i['problem'] for i in problems_db], convert_to_tensor=True).to(device)
    sim_mat = util.cos_sim(q_embs, p_embs)
    top1 = torch.max(sim_mat, dim=1)[0]
    return top1.mean().item()


class NestedContrastiveWithNoisePenalty(nn.Module):
    def __init__(self, model: SentenceTransformer,
                 lambda_noise: float = 0.5,
                 margin: float = 0.3):
        super().__init__()
        self.model = model
        self.contrastive = losses.OnlineContrastiveLoss(
            model=model,
            distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE,
            margin=margin
        )
        self.lambda_noise = lambda_noise

    def forward(self, sentence_features, labels,
                noise_queries=None, problems_db=None, device=None):
        c_loss = self.contrastive(sentence_features, labels)
        if noise_queries and problems_db:
            noise_sim = eval_noise_similarity(self.model, noise_queries, problems_db, device)
            noise_loss = torch.tensor(noise_sim ** 2, device=c_loss.device)
        else:
            noise_loss = torch.tensor(0.0, device=c_loss.device)
        return c_loss + self.lambda_noise * noise_loss, c_loss.detach(), noise_loss.detach()



if __name__ == "__main__":
    model_name = "bge-m3"
    batch_size = 8
    lr = 5e-6
    max_epochs = 1
    eval_every_steps = 100
    early_stop_thresh = 0.90

    problems_db, clean_inputs = load_data()
    daily_neg_queries = load_daily_dataset()

    model = SentenceTransformer(model_name, device=device)
    for param in model._first_module().auto_model.embeddings.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    nested_loss = NestedContrastiveWithNoisePenalty(
        model=model, lambda_noise=0.5, margin=0.3
    ).to(device)

    saved_snapshots = None
    global_step = 0
    stop_flag = False
    prev_noise_sim = 0


    train_examples = build_dataset_dynamic(
        model, problems_db, clean_inputs,
        neg_queries=daily_neg_queries, num_neg=100, hard_pool=250,
        use_neg=True, manual_done=True
    )
    train_loader = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size,
        collate_fn=model.smart_batching_collate
    )

    # —— 将训练集保存到本地，供 Stage1 前检查 ——
    os.makedirs('data', exist_ok=True)
    # 数据集以三元组形式保存：[(text_a, text_b, label), ...]
    serializable = [[ex.texts[0], ex.texts[1], ex.label] for ex in train_examples]
    with open('data/answer2problem.json', 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logging.info("Saved initial train dataset (triples) to data/answer2problem.json")

