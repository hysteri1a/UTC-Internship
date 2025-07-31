# finetune.py

# ---------- 1. data_annotation.py ----------
import os
import json
import tempfile
import base64
from PIL import Image
from openai import OpenAI

# 配置
RAW_DIR = 'raw_images'
ANNOTATED_DIR = 'annotated_samples'
META_FILE = 'error_samples/meta.jsonl'
API_BASE = 'http://192.168.1.224:8002/v1'
MODEL_NAME = 'qwen-omni'

os.makedirs(ANNOTATED_DIR, exist_ok=True)

# 编码图片为 base64
def encode_image(path):
    with open(path, 'rb') as f:
        data = f.read()
        return base64.b64encode(data).decode('utf-8')

# 初始化 OpenAI 客户端
client = OpenAI(api_key='EMPTY', base_url=API_BASE)

with open(META_FILE, 'w', encoding='utf-8') as meta_f:
    for fname in os.listdir(RAW_DIR):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        src_path = os.path.join(RAW_DIR, fname)
        # 预处理：可选，调用本地增强脚本
        # e.g., run preprocessing_pipeline on this image first

        # 调用模型预测
        b64 = encode_image(src_path)
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': '你是一个智能助手.'},
                {'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                    {'type': 'text', 'text': '你是一个专业的手写文字识别专家。请仔细识别这张图片中的所有手写文本,即使笔迹潦草或部分模糊也要尽力识别。按原样输出文本内容，不要修改任何字符。如果无法确定某个字符，用\'□\'代替。输出格式：\n识别结果：<文本内容>\")函数。'}
                ]}
            ]
        )
        pred = resp.choices[0].message.content.strip()

        # 展示给用户：模型预测 vs 需要标注
        img = Image.open(src_path)
        img.show()
        print(f"模型预测: {pred}")
        label = input(f"请判断模型预测与实际是否相符？相符输入1; 不符输入0: ")

        # 判断是否为错误样本
        is_error = (label != "1")
        if not is_error:
            print("预测正确，跳过。")
            img.close()
            continue
        else:
            label = input(f"请输入正确的内容: ")
            while input("请确认你的输入正确性: 1正确;2不正确") != "1":
                label = input(f"请输入正确的内容: ")
            # 归档和记录
            dst_name = f"char_{label}_{fname}"
            dst_path = os.path.join(ANNOTATED_DIR, dst_name)
            os.rename(src_path, dst_path)
            meta_f.write(json.dumps({'image_path': dst_path, 'label': label}, ensure_ascii=False) + '\n')
            img.close()
            print(f"记录错误样本: {dst_name}")

print(f"标注完成，meta 存于 {META_FILE}")

# ---------- 2. sample_collection.py ----------
import os
import json
import random
from PIL import Image, ImageFilter

DATA_DIR = 'error_samples/annotated_samples'
OUTPUT_DIR = 'error_samples/augmented'
META_FILE = 'error_samples/meta.jsonl'

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(META_FILE, 'r', encoding='utf-8') as f:
    samples = [json.loads(l) for l in f]

def augment_image(src, dst):
    img = Image.open(src).convert('L')
    img = img.rotate(random.uniform(-5,5), expand=True, fillcolor=255)
    if random.random()<0.5:
        img = img.filter(ImageFilter.GaussianBlur(1))
    img.save(dst)

aug_meta = []
for s in samples:
    name, ext = os.path.splitext(os.path.basename(s['image_path']))
    for i in range(2):
        out = os.path.join(OUTPUT_DIR, f"{name}_aug{i}{ext}")
        augment_image(s['image_path'], out)
        aug_meta.append({'image_path': out, 'label': s['label']})

with open(META_FILE, 'a', encoding='utf-8') as f:
    for rec in aug_meta:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
print(f"已生成 {len(aug_meta)} 条增强样本")

# ---------- 3. finetune_qwen_omni.py ----------
from transformers import Trainer, TrainingArguments, AutoModelForImageClassification
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

MODEL_NAME = 'qwen2.5-omni'
OUTPUT_DIR = 'ft_qwen_omni_errorfix'

ds = load_dataset('json', data_files={'train':'error_samples/meta.jsonl'})
transform = Compose([Resize((224,224)), ToTensor(), Normalize([0.5],[0.5])])
def prep(ex):
    pix = [transform(Image.open(p).convert('RGB')) for p in ex['image_path']]
    return {'pixel_values':pix,'labels':ex['label']}
ds = ds.with_transform(prep)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=len(ds['train'].features['labels'].names)
)
args = TrainingArguments(output_dir=OUTPUT_DIR, per_device_train_batch_size=8,
    learning_rate=3e-6, num_train_epochs=4, lr_scheduler_type='cosine',save_strategy='epoch')
trainer = Trainer(model, args, train_dataset=ds['train'])
trainer.train(); trainer.save_model(OUTPUT_DIR)

# ---------- 4. preprocessing_pipeline.py ----------
import cv2, numpy as np, os
INPUT='raw_images'; OUTPUT='preprocessed_images'; os.makedirs(OUTPUT,exist_ok=True)
for f in os.listdir(INPUT):
    if not f.lower().endswith(('png','jpg','jpeg')): continue
    img=cv2.imread(os.path.join(INPUT,f),cv2.IMREAD_GRAYSCALE)
    img=cv2.medianBlur(img,3)
    img=cv2.morphologyEx(img,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)))
    edges=cv2.Canny(img,50,150)
    closed=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    skel=np.zeros_like(closed)
    _,bin=cv2.threshold(closed,127,255,0)
    elem=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    while cv2.countNonZero(bin)>0:
        er=cv2.erode(bin,elem); dt=cv2.dilate(er,elem)
        skel=cv2.bitwise_or(skel,cv2.subtract(bin,dt)); bin=er
    clahe=cv2.createCLAHE(2.0,(8,8)).apply(img)
    adapt=cv2.adaptiveThreshold(clahe,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    out=cv2.bitwise_or(skel,adapt)
    out=cv2.resize(out,(224,224))
    cv2.imwrite(os.path.join(OUTPUT,f),out)
print(f"预处理完成: {OUTPUT}")
