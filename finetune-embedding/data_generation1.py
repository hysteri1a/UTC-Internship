import json
import re
import random
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, util, losses, InputExample
import numpy as np
import os

# —— 日志配置 ——
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# —— 选择 GPU ——
GPU_ID = 3
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
        lst = list(text); lst[idx], lst[idx+1] = lst[idx+1], lst[idx]
        return ''.join(lst)
    if op == 'delete': return text[:idx] + text[idx+1:]
    if op == 'dup':    return text[:idx] + text[idx] + text[idx:]
    return text


def load_daily_dataset(path="../data/modified_daily_datasets.json", max_samples=100):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    queries=[]
    for item in data:
        for msg in item.get('conversation',[]):
            inp=msg.get('input')
            if inp: queries.append(inp.strip())
    queries=list(set(queries)); random.shuffle(queries)
    return queries[:max_samples]


def load_data(json_qna_path="../data/parsed_data.json", json_conv_path="../data/output.json"):
    # 读取 QnA 库 和 原始对话，分组返回
    with open(json_qna_path, encoding='utf-8') as f:
        qna=json.load(f)
    problems_db=[{'problem':i['problems'],'answer':i.get('answer','')} for i in qna]
    with open(json_conv_path, encoding='utf-8') as f:
        convs=json.load(f)
    grouped=[]
    for conv in convs:
        msgs=[]
        for msg in conv.get('conversation',[]):
            inp=re.sub(r"(客户已接入，会话开始。|会话已超时结束。|\[图片\])","",msg.get('input','')).strip()
            out=msg.get('output','').strip()
            if inp and out: msgs.append((inp,out))
        if msgs: grouped.append(msgs)
    return problems_db,grouped


def annotate_positive_candidate(model, query, problems_db, threshold=0.3, ans_threshold=0.79):
    """
    Show up to 5 highest‐similarity answers ≥ ans_threshold for manual labeling.
    Prints each with its similarity score.
    Returns:
      - {'positive': {...}, 'negatives': [...]} when a choice 1–N is made
      - 'STOP' if user enters 'q'
      - {'positive': None, 'negatives': [...]} if user enters anything else (treat all candidates as negatives)
    """
    inp_text, out_text = query
    # compute similarities
    resp_emb = model.encode(out_text, convert_to_tensor=True)
    ans_embs = model.encode([item['answer'] for item in problems_db], convert_to_tensor=True)
    sims = util.cos_sim(resp_emb, ans_embs)[0].cpu().numpy()

    # take top 5 overall
    top_idxs = list(np.argsort(-sims)[:5])
    # filter by answer threshold
    candidates = [idx for idx in top_idxs if sims[idx] >= ans_threshold]
    if not candidates:
        print(f"No candidates above ans_threshold {ans_threshold}, skip annotation.")
        return 10

    # print for annotation
    print("=== Response Annotation ===")
    print(f"Response: {out_text}")
    for i, idx in enumerate(candidates, start=1):
        print(f"{i}. [{sims[idx]:.4f}] Answer: {problems_db[idx]['answer']} | Q: {problems_db[idx]['problem']}")
    print(f"{len(candidates)+1}. None of the above")

    choice = input("Your choice: ").strip().lower()
    if choice == 'q':
        return 'STOP'
    if choice.isdigit():
        c = int(choice)
        if 1 <= c <= len(candidates):
            sel_idx = candidates[c-1]
            negs = [problems_db[i] for i in candidates if i != sel_idx]
            return {'positive': problems_db[sel_idx], 'negatives': negs}

    # anything else: treat all candidates as negatives
    negs = [problems_db[idx] for idx in candidates]
    return {'positive': None, 'negatives': negs}


def generate_random_negatives(inp_emb,problem_embs,problems_db,num_neg=20,hard_pool=50,neg_queries=None,use_neg=False,pos_threshold=0.8):
    sims=util.cos_sim(inp_emb,problem_embs)[0].cpu().numpy()
    idxs=np.argsort(-sims); max_idx=idxs[0]; max_sim=sims[max_idx]
    samples=[]
    if max_sim>=pos_threshold: samples.append((problems_db[max_idx],1.0))
    pool=idxs[1:hard_pool+1].tolist();
    for i in random.sample(pool,min(num_neg,len(pool))): samples.append((problems_db[i],0.0))
    if use_neg and neg_queries: samples.append(({'problem':random.choice(neg_queries),'answer':''},0.0))
    return samples


def build_dataset_dynamic(model,problems_db,grouped_convs,neg_queries=None,num_neg=100,hard_pool=200,use_neg=False,manual_done=False,ans_threshold=0.79):
    """
    manual_done=False 时：
      - sel==10 累 prefix
      - 选正/负 或 STOP 重置 prefix
    manual_done=True 后：按 prefix+inp 自动标注
    """
    db_ans=[i['answer'] for i in problems_db]
    ans_embs=model.encode(db_ans,convert_to_tensor=True)
    examples=[]
    prefix=""
    for conv in grouped_convs:
        prefix=""
        for inp,out in conv:
            if not manual_done:
                sel = annotate_positive_candidate(model,(inp,out),problems_db,ans_threshold=ans_threshold)
                prefix += inp
                if sel=='STOP':
                    manual_done=True
                    prefix=""
                    break
                if sel==10:
                    continue
                if sel.get('positive'):
                    examples.append(InputExample(texts=[prefix,sel['positive']['problem']],label=1.0))
                    print(prefix)
                    print(sel['positive']['problem'])
                    examples.append(InputExample(texts=[augment_text(prefix),sel['positive']['problem']],label=1.0))
                    for neg in sel['negatives']:
                        examples.append(InputExample(texts=[prefix, neg['problem']], label=0.0))
                    prefix = ""
                    continue
                for neg in sel['negatives']:
                    examples.append(InputExample(texts=[prefix,neg['problem']],label=0.0))
                continue
            # 自动标注
            query=prefix+inp if prefix else inp
            q_emb=model.encode(query,convert_to_tensor=True)
            sims=util.cos_sim(q_emb,ans_embs)[0].cpu().numpy()
            best=int(np.argmax(sims))
            if sims[best]>=ans_threshold:
                examples.append(InputExample(texts=[query,problems_db[best]['problem']],label=1.0))
                examples.append(InputExample(texts=[augment_text(query),problems_db[best]['problem']],label=1.0))
            for p,l in generate_random_negatives(q_emb,ans_embs,problems_db,num_neg,hard_pool,neg_queries,use_neg):
                examples.append(InputExample(texts=[query,p['problem']],label=l))
            prefix=""
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
    early_stop_thresh = 0.70

    problems_db, clean_inputs = load_data()
    daily_neg_queries = load_daily_dataset()

    model = SentenceTransformer(model_name, device=device)
    for param in model._first_module().auto_model.embeddings.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    nested_loss = NestedContrastiveWithNoisePenalty(
        model=model, lambda_noise=0.5, margin=0.3
    ).to(device)

    train_examples = build_dataset_dynamic(
        model, problems_db, clean_inputs,
        neg_queries=None, num_neg=100, hard_pool=250, use_neg=False, manual_done=False
    )
    train_loader = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size,
        collate_fn=model.smart_batching_collate
    )

    # —— 将训练集保存到本地，供 Stage1 前检查 ——
    os.makedirs('data', exist_ok=True)
    # 数据集以三元组形式保存：[(text_a, text_b, label), ...]
    serializable = [[ex.texts[0], ex.texts[1], ex.label] for ex in train_examples]
    with open('data/annonated_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logging.info("Saved initial train dataset (triples) to data/annonated_dataset.json")



