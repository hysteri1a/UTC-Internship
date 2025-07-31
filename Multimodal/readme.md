# 多模态 AI 工具链 API 使用说明（含调用示例）

该 API 提供了一套基于 LangGraph 和 Qwen 模型的多模态 AI 工具链，支持图像识别、音频处理、文档分析和模型微调。所有数据交互均使用 JSON 格式。

## ⚙️ 环境准备

创建独立 Python 环境：
```bash
python -m venv .venv
# 或
conda create -n multimodal python=3.10
```

安装依赖：
```bash
pip install requirements.txt
```

代码位置：项目根目录

## 0. 启动服务

在服务器上启动模型服务：
```
# 启动 Qwen 模型服务

# 启动多模态服务  

```

## 1. main.py - LangGraph 工具链执行

**功能**：基于 LangGraph 构建的对话式 AI 系统，支持自动工具调用和结果整合。

**说明**：主程序入口，连接 Qwen 模型服务，注册多模态工具和文档工具，构建智能对话流。

**运行示例**：
```bash
python main.py
```

**工具链配置**：
```python
# 连接本地 Qwen 模型
chat_model = ChatOpenAI(
    base_url="http://192.168.1.224:8000/v1",
    api_key="EMPTY",
    model="qwen3-32b",
).bind_tools([multi_tool, word_tool])  # 绑定工具（可自定义）

# 注册工具节点
tool_node = ToolNode([multimodal_tool, word_tool])

# 构建对话图 意味着有连线的工具可以互相调用
graph = MessageGraph()
graph.add_node("chat", chat_model)
graph.add_node("tools", tool_node)
```

**调用细节**：
```python
# 多模态推理调用
initial_messages = [
    HumanMessage(content=(
        "请调用 multimodal_inference(image_path=\"./temp/test18.png\", "
        "audio_path=\" \", prompt=\"请识别图片,告诉我这个图片是干什么的。\")函数。"
    ))
]
final_state = app.invoke(initial_messages)

# Word 文档分析调用
initial = [HumanMessage(content=(
    '请调用 word_tool(file_path="./test.docx", '
    'prompt="请总结文档的主要内容并提取图表信息")'
))]
final = app.invoke(initial)
```

**返回格式**：
```python
# 查看结果
for msg in final_state:
    if isinstance(msg, AIMessage):
        print("[AI]", msg.content.strip())
    elif isinstance(msg, ToolMessage):
        print("[Tool Output]", msg.content.strip())
```

## 2. combined_tool.py - 多模态处理核心

**功能**：提供多模态推理和文档分析的核心功能实现。

**说明**：包含图像识别、音频处理、Word 文档解析等底层处理逻辑，支持本地和远程模型调用。

### 2.1 多模态推理接口

**函数**：`multimodal_inference(image_path, text_prompt, max_new_tokens)`

**参数**：
- `image_path` (str): 图片文件路径
- `text_prompt` (str): 文字提示，默认"请描述这张图片"
- `max_new_tokens` (int): 生成文本最大token数，默认256

**调用示例**：
```python
from combined_tool import multimodal_inference

text_result, audio_result = multimodal_inference(
    image_path="./temp/test18.png",
    text_prompt="请识别图片，告诉我这个图片是干什么的"
)
```

**返回**：
```python
(
    "这是一张显示数据分析图表的图片，包含折线图和柱状图...",  # 文本分析结果
    "音频内容分析结果..."  # 音频分析结果
)
```

### 2.2 文档分析接口

**函数**：`extract_from_word(doc_path, prompt)`

**参数**：
- `doc_path` (str): Word 文档路径
- `prompt` (str): 分析任务的提示词

**调用示例**：
```python
from combined_tool import extract_from_word

result = extract_from_word(
    doc_path="./test.docx",
    prompt="请总结文档的主要内容并提取图表信息"
)
```

**返回**：
```text
请总结文档的主要内容并提取图表信息

文档内容:
第一章 项目概述
本项目旨在...

第二章 技术方案
采用的技术包括...

提取到图表图片: temp_chart_0.png
提取到图表图片: temp_chart_1.png
```

**支持功能**：
- Word 文档文本提取
- 嵌入图片/图表识别和保存
- 文档结构解析
- 智能内容分析

## 3. tools.py - LangChain 工具定义

**功能**：定义 LangChain 工具接口，封装核心功能为标准化工具。

**说明**：使用 Pydantic 定义输入 schema，通过 @tool 装饰器创建可被 LangGraph 调用的工具。

### 3.1 多模态工具定义(工具的一个实现)

**工具名**：`multimodal_inference`

**Schema 定义**：
```python
class MultiInput(BaseModel):
    image_path: str = Field(..., description="本地图片文件路径或 URL")
    audio_path: str = Field(..., description="本地音频文件路径或 URL")
    prompt: str = Field(..., description="对图音信息的文字提示，如"请描述..."")

@tool("multimodal_inference", description="同时处理图片和音频，并根据提示生成综合回答", args_schema=MultiInput)
def multimodal_tool(image_path: str, audio_path: str, prompt: str) -> str:
    from combined_tool import multimodal_inference
    return multimodal_inference(image_path, audio_path, prompt)
```

### 3.2 文档工具定义(工具的一个实现)

**工具名**：`word_tool`

**Schema 定义**：
```python
class WordToolInput(BaseModel):
    file_path: str = Field(..., description="Word 文件路径")
    prompt: str = Field(..., description="提取文档内容的提示词")

@tool("word_tool", description="读取 Word 文档，并结合提示词进行总结与图表提取", args_schema=WordToolInput)
def word_tool(file_path: str, prompt: str) -> str:
    from combined_tool import extract_from_word
    return extract_from_word(file_path, prompt)
```

**工具包装**：
```python
# 绑定为 LangGraph Tool
multi_tool = Tool(name="multimodal_inference", func=multimodal_tool, description="一次性处理图片和音频，返回综合回答", args_schema=MultiInput)
word_tool_wrapper = Tool(name="word_tool", func=word_tool, description="用于分析 Word 文档内容和图表", args_schema=WordToolInput)
```

## 4. add_tool.py - 工具自动添加脚本

**功能**：自动化工具添加脚本，快速扩展新功能模块到 tools.py。

**说明**：通过命令行参数生成标准化的工具定义代码，支持自定义工具名称、描述和调用函数。

**使用示例**：
```bash
python add_tool.py \
  --tool-name image_caption_tool \
  --tool-desc "图像字幕生成工具" \
  --input-desc "图片文件路径" \
  --module-name image_model \
  --call-fn predict_caption \
  --tools-file tools.py
```

**参数说明**：
- `--tool-name`: 工具函数名（如 image_caption_tool）
- `--tool-desc`: @tool 装饰器的描述文字
- `--input-desc`: 输入 path 字段的描述
- `--module-name`: 调用的模块名（默认 image_model）
- `--call-fn`: 调用的函数名（默认 predict_image_caption）
- `--tools-file`: 目标 tools.py 文件路径

**生成的工具代码**：
```python
class ImageCaptionTool(BaseModel):
    path: str = Field(..., description="图片文件路径")

@tool("image_caption_tool", description="图像字幕生成工具", args_schema=ImageCaptionTool)
def image_caption_tool(path: str) -> str:
    from image_model import predict_caption
    return predict_caption(path)
```

**代码模板**：
```python
TEMPLATE = '''
class {class_name}(BaseModel):
    path: str = Field(..., description="{input_desc}")

@tool("{tool_name}", description="{tool_desc}", args_schema={class_name})
def {tool_name}(path: str) -> str:
    from {module_name} import {call_fn}
    return {call_fn}(path)
'''
```

## 存储结构

项目文件组织结构：

```
./
├── main.py                    # LangGraph 工具链主程序
├── combined_tool.py           # 多模态处理核心功能
├── tools.py                   # LangChain 工具定义
├── finetune.py               # 模型微调完整流程
├── add_tool.py               # 工具自动添加脚本
├── raw_images/               # 原始图片数据
├── preprocessed_images/      # 预处理后图片
├── error_samples/            # 错误样本收集
│   ├── annotated_samples/    # 标注样本
│   ├── augmented/           # 增强样本
│   └── meta.jsonl           # 样本元数据
├── ft_qwen_omni_errorfix/   # 微调模型输出
└── temp/                    # 临时文件目录
    ├── test18.png
    ├── audio.wav
    └── temp_chart_*.png
```

## 特性说明

### 🎯 多模态理解 (combined_tool.py)
- **图像识别**：支持场景理解、物体检测、文字识别
- **音频处理**：语音识别、音频分析、情感识别
- **多模态融合**：图像和音频信息的综合分析
- **8-bit 量化**：BitsAndBytesConfig 优化显存使用

### 📄 文档智能分析 (combined_tool.py)
- **格式支持**：Word 文档完整解析
- **内容提取**：文本、图表、表格全面提取
- **图表处理**：自动提取嵌入图片并保存为临时文件
- **结构识别**：段落层次和文档结构分析

### 🔧 工具链架构 (main.py + tools.py)
- **模块化设计**：独立的工具组件，易于扩展
- **流程编排**：LangGraph 构建的智能对话流
- **标准化接口**：Pydantic schema 定义的统一输入格式
- **状态管理**：多轮对话的上下文保持

## 🔧 配置参数

### 模型服务配置
```python
# Qwen 模型服务地址
BASE_URL = "http://192.168.1.224:8000/v1"
OMNI_BASE_URL = "http://192.168.1.224:8002/v1"

# 本地模型路径
MODEL_DIR = "../../../Model/qwen-omni"

# GPU 显存分配
max_mem = {
    0: "0GB",    # 物理 GPU0 - 不使用
    1: "0GB",    # 物理 GPU1 - 不使用
    2: "23GB",   # 物理 GPU2 - 主要推理卡
    3: "23GB",   # 物理 GPU3 - 备用推理卡
}
```
