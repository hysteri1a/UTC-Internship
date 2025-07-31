# combined_tool.py

from io import BytesIO
import os
import torch
import numpy as np
from PIL import Image
import soundfile as sf
import docx
from docx.opc.constants import RELATIONSHIP_TYPE as RT

import os
import torch
import soundfile as sf
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from transformers import BitsAndBytesConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import torch

# 1. 安装 bitsandbytes（如果还没装的话）
#    pip install bitsandbytes

# 2. 定义 8-bit Quantization 配置
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,      # 启用 8-bit 量化加载
    llm_int8_threshold=6.0, # （可选）控制何时切换到 8-bit；一般默认即可
    llm_int8_has_fp16_weight=False  # 如果想让部分权重保持 fp16，可设为 True
)

# 模型路径设置
MODEL_DIR = "../../../Model/qwen-omni"  # 本地模型路径
# MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"  # 或者使用在线模型
# 如果你之前设置过 CUDA_VISIBLE_DEVICES，先清除它
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# 明确暴露全部 4 张卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 可省略，默认就是全部可见

# 定义每张卡的显存上限
max_mem = {
    0: "0GB",    # 物理 GPU0
    1: "0GB",    # 物理 GPU1
    2: "23GB",   # 物理 GPU2
    3: "23GB",   # 物理 GPU3
}


# 加载多模态处理器
processor = Qwen2_5OmniProcessor.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True
)


import base64
from io import BytesIO
import requests
import soundfile as sf
from PIL import Image
import numpy as np
import librosa


def process_mm_info(conversation, use_audio_in_video=False):
    """
    处理多模态信息的辅助函数，从对话中提取音频、图像和视频信息。
    支持新格式：image_url, audio_url, input_audio (Base64)。
    """
    audios = []
    images = []
    videos = []

    for message in conversation:
        if not isinstance(message, dict) or "content" not in message:
            continue

        content_list = message["content"]
        if not isinstance(content_list, list):
            continue

        for content in content_list:
            if not isinstance(content, dict) or "type" not in content:
                continue

            content_type = content["type"]

            # 处理图像 (支持 image 和 image_url)
            if content_type in ["image", "image_url"]:
                image_key = "image" if content_type == "image" else "image_url"
                image_data = content.get(image_key, {})
                if isinstance(image_data, dict):
                    image_path = image_data.get("url", "")
                else:
                    image_path = image_data

                if image_path:
                    try:
                        if image_path.startswith("http"):
                            response = requests.get(image_path)
                            image = Image.open(BytesIO(response.content)).convert("RGB")
                        else:
                            image = Image.open(image_path).convert("RGB")
                        images.append(image)
                        print(f"Processed image: {image_path}")
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")

            # 处理音频 (支持 audio 和 audio_url)
            elif content_type in ["audio", "audio_url"]:
                audio_key = "audio" if content_type == "audio" else "audio_url"
                audio_data = content.get(audio_key, {})
                if isinstance(audio_data, dict):
                    audio_path = audio_data.get("url", "")
                else:
                    audio_path = audio_data

                if audio_path:
                    try:
                        if audio_path.startswith("http"):
                            response = requests.get(audio_path)
                            audio_data = BytesIO(response.content)
                            waveform, sr = sf.read(audio_data)
                        else:
                            waveform, sr = sf.read(audio_path)

                        target_sr = getattr(processor.feature_extractor, 'sampling_rate', 24000)
                        if sr != target_sr:
                            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
                        audios.append(np.asarray(waveform, dtype=np.float32))
                        print(f"Processed audio: {audio_path}")
                    except Exception as e:
                        print(f"Error processing audio {audio_path}: {e}")

            # 处理 Base64 编码音频 (input_audio)
            elif content_type == "input_audio":
                audio_data = content.get("input_audio", {}).get("data", "")
                if audio_data:
                    try:
                        # 解码 Base64
                        waveform = base64.b64decode(audio_data)
                        waveform, sr = sf.read(BytesIO(waveform))
                        target_sr = getattr(processor.feature_extractor, 'sampling_rate', 24000)
                        if sr != target_sr:
                            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
                        audios.append(np.asarray(waveform, dtype=np.float32))
                        print("Processed Base64 audio")
                    except Exception as e:
                        print(f"Error processing Base64 audio: {e}")

            # 处理文本
            elif content_type == "text":
                print(f"Text content: {content.get('text', '')[:50]}...")

    return audios, images, videos


def wrap_image_as_conversation(image_path: str, text_prompt: str = None):
    """
    接受图片路径，返回标准格式的 conversation 列表。
    text_prompt 可选，附加在图片后的文字提问。
    """
    content = [{"type": "image", "image": image_path}]
    if text_prompt:
        content.append({"type": "text", "text": text_prompt})

    return [
        {
            "role": "user",
            "content": content
        }
    ]


from openai import OpenAI

def encode_image(image_path):       # 编码本地图片的函数
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_response(text_prompt, image_path=None, audio_path=None):  #test.jpg
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://192.168.1.224:8002/v1",
    )
    text=""
    audio=""
    if image_path:
        base64_image = encode_image(image_path)  # 编码本地图片
        print('encoding_image:', base64_image)
        completion = client.chat.completions.create(
            model="qwen-omni",  # 或根据你环境设置
            messages=[
                {"role": "system", "content": "你是一个智能助手."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        },
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
        )
        text = completion.choices[0].message.content
    if audio_path:
        base64_image = encode_image(audio_path)  # 编码本地图片
        print('encoding_image:', base64_image)
        completion = client.chat.completions.create(
            model="qwen-omni",
            messages=[
                {"role": "system", "content": "你是一个智能助手."},
                {"role": "user",
                 "content": [
                     {
                         "type": "audio_url","audio_url": {"url": f"data:audio/wav;base64,{base64_image}"}
                     },
                     {"type": "text", "text": text_prompt}
                 ]
                 }
            ]
        )

        audio = completion.choices[0].message.content

    # 输出结果
    print("text:", text)
    print("audio:", audio)
    return text,audio


def multimodal_inference(image_path, text_prompt="请描述这张图片", max_new_tokens=256):
    """
    处理图片路径并生成相关文本的函数

    参数:
    - image_path: 图片的本地路径 (str)
    - text_prompt: 可选的文本提示，默认为"请描述这张图片" (str)
    - max_new_tokens: 生成文本的最大token数 (int)

    返回:
    - generated_text: 生成的文本描述 (str)
    """
    text, audio = get_response(text_prompt, image_path=image_path)

    return text, audio


def extract_from_word(doc_path: str, prompt: str) -> str:
    """
    从 Word 文档中提取文本和图表，并结合提示生成摘要。
    """
    if not os.path.exists(doc_path):
        return f"文件 {doc_path} 不存在"

    try:
        doc = docx.Document(doc_path)
        full_text = [para.text for para in doc.paragraphs]
        content = "\n".join(full_text)

        # 提取嵌入的图片作为图表
        chart_descriptions = []
        rels = doc.part._rels
        for idx, rel in enumerate(rels):
            r = rels[rel]
            if r.reltype == RT.IMAGE:
                image_data = r.target_part.blob
                image = Image.open(BytesIO(image_data)).convert("RGB")
                temp_path = f"temp_chart_{idx}.png"
                image.save(temp_path)
                chart_descriptions.append(f"提取到图表图片: {temp_path}")

        return (
            f"{prompt}\n\n文档内容:\n{content}\n\n" +
            "\n".join(chart_descriptions)
        )
    except Exception as e:
        return f"处理 Word 文件时出错: {str(e)}"


