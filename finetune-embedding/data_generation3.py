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
GPU_ID = 2
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
    读取 QnA 数据并返回：
      - problems_db: [{'problem','answer'}, ...]
      - grouped_convs: list of conversations, each is [(txt, out), ...]
    """
    # 加载问题库
    with open(json_qna_path, encoding="utf-8") as f:
        qna = json.load(f)
    problems_db = [{'problem': item['problems'], 'answer': item.get('answer', '')}
                   for item in qna]

    # 加载对话并按 conversation 分组
    with open(json_conv_path, encoding="utf-8") as f:
        convs = json.load(f)
    grouped_convs = []
    for conv in convs:
        msgs = []
        for msg in conv.get('conversation', []):
            inp = msg.get('input', '').strip()
            out = msg.get('output', '').strip()
            # 去除开头、超时、图片占位
            txt = re.sub(r"(客户已接入，会话开始。|会话已超时结束。|\[图片\])", "", inp).strip()
            if txt and out:
                msgs.append((txt, out))
        if msgs:
            grouped_convs.append(msgs)

    return problems_db, grouped_convs


def generate_random_negatives(inp_emb, problem_embs, problems_db,
                              num_neg=20, hard_pool=50,
                              neg_queries=None, use_neg=False,
                              pos_threshold=0.82):
    """
    基于 inp_emb 和库中 problem_embs
      - 若最高相似度 >= pos_threshold，添加该例为正样本
      - 随机采 num_neg 个次高相似度样本为负样本
      - 可选添加一个随机 daily_neg_queries 作为噪声负样本
    返回 List[(problem_dict, label_float)]
    """
    sims = util.pytorch_cos_sim(inp_emb, problem_embs)[0].cpu().numpy()
    desc_idx = np.argsort(-sims)
    max_idx = desc_idx[0]
    max_sim = sims[max_idx]

    samples = []
    # 正样本
    if max_sim >= pos_threshold:
        samples.append((problems_db[max_idx], 1.0))
    # 负样本
    neg_pool = desc_idx[1: hard_pool+1].tolist()
    neg_count = min(num_neg, len(neg_pool))
    for idx in random.sample(neg_pool, neg_count):
        samples.append((problems_db[idx], 0.0))
    # 噪声负样本
    if use_neg and neg_queries:
        neg_q = random.choice(neg_queries)
        samples.append(({'problem': neg_q, 'answer': ''}, 0.0))

    return samples


def build_dataset_dynamic(model, problems_db, grouped_convs,
                          neg_queries=None,
                          num_neg=100, hard_pool=200,
                          use_neg=False,
                          ans_threshold=0.82,
                          batch_size=64):
    """
    批量预编码所有 prefix 与 out_text，避免重复 encode。
    """
    # 1) 构造所有 prefix 与 out_text 列表
    prefixes = []
    out_texts = []
    for conv in grouped_convs:
        prefix = ""
        for txt, out in conv:
            prefix += txt
            prefixes.append(prefix)
            out_texts.append(out)

    # 2) 预编码库答案、prefixes、out_texts
    db_answers = [item['answer'] for item in problems_db]
    ans_embs    = model.encode(db_answers,  batch_size=batch_size, convert_to_tensor=True)
    prefix_embs = model.encode(prefixes,   batch_size=batch_size, convert_to_tensor=True)
    out_embs    = model.encode(out_texts,  batch_size=batch_size, convert_to_tensor=True)

    examples = []
    ptr = 0
    # 3) 遍历每个 prefix/out pair
    for conv in grouped_convs:
        for txt, out in conv:
            prefix_emb = prefix_embs[ptr]
            out_emb    = out_embs[ptr]
            ptr += 1

            sims = util.cos_sim(out_emb, ans_embs)[0].cpu().numpy()
            max_idx = int(np.argmax(sims))
            max_sim = sims[max_idx]

            prefix = prefixes[ptr-1]
            if max_sim < ans_threshold:
                negs = generate_random_negatives(
                    prefix_emb, ans_embs, problems_db,
                    num_neg, hard_pool, neg_queries, use_neg
                )
                for prob, lbl in negs:
                    examples.append(InputExample(texts=[prefix, prob['problem']], label=lbl))
            else:
                # 正样本
                pos_problem = problems_db[max_idx]['problem']
                query = prefix
                examples.append(InputExample(texts=[query, pos_problem], label=1.0))
                examples.append(InputExample(texts=[augment_text(query), pos_problem], label=1.0))
                # 负样本
                negs = generate_random_negatives(
                    model.encode(query, convert_to_tensor=True),
                    ans_embs, problems_db,
                    num_neg, hard_pool, neg_queries, use_neg
                )
                for prob, lbl in negs:
                    examples.append(InputExample(texts=[query, prob['problem']], label=lbl))

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
    lr = 5e-6
    max_epochs = 1
    early_stop_thresh = 0.90

    problems_db, grouped_convs = load_data()
    daily_neg_queries = load_daily_dataset()

    model = SentenceTransformer(model_name, device=device)
    # 冻结 embedding 层
    for param in model._first_module().auto_model.embeddings.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    nested_loss = NestedContrastiveWithNoisePenalty(model).to(device)

    # 构建并保存初始训练集
    train_examples = build_dataset_dynamic(
        model, problems_db, grouped_convs,
        neg_queries=daily_neg_queries,
        num_neg=100, hard_pool=250,
        use_neg=True, 
        ans_threshold=0.82
    )
    os.makedirs('data', exist_ok=True)
    serializable = [[ex.texts[0], ex.texts[1], ex.label] for ex in train_examples]
    with open('data/answer2problem1.json', 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logging.info("Saved initial train dataset to data/answer2problem1.json")
