# 智能对话管道系统 (Intelligent Chat Pipeline System)

## 概述

这是一个基于langchain的智能对话管道系统，集成了多模态文档检索、语义理解、工具调用和大语言模型回复生成功能。系统采用管道架构设计，支持复杂的多轮对话和智能意图识别。

## 系统架构

```
用户输入 → 语义分析 → 触发器匹配 → 变量填充 → 并行处理 → 回复生成
                                              ↓
                                    [文档检索 + API调用]
```

## 核心组件

### 1. ChatManager - 对话管理器

负责维护对话历史和生成AI回复的核心组件。

**主要功能：**
- 消息历史管理（SystemMessage, HumanMessage, AIMessage）
- 自动消息裁剪（防止token溢出）
- 集成HuggingFace本地大语言模型
- 支持多种输入格式的回复生成

**关键方法：**
```
def __init__(self, system_content: str, max_tokens: int = 1000, model_path="")
def add_human_message(self, content: str)
def add_ai_message(self, human_message, prompt=None, retriever_info=None, database_info=None, api_prompt=None)
```

### 2. MultiRetrieverManager - 多路检索管理器

实现多种检索策略的并行文档检索系统。

**支持的检索方式：**
- BM25关键词检索（精确匹配）
- FAISS语义向量检索（语义相似度）
- 可扩展添加更多检索器

**特性：**
- 异步并行检索
- 自动文档去重
- 相似度阈值过滤

**关键方法：**
```python
def add_retriever(self, retriever: BaseRetriever, embedding: Optional[Embeddings] = None)
async def retrieve(self, query: str) -> List[Document]
```

### 3. ExampleSelector - 示例选择器

基于语义相似度的智能示例匹配系统。

**选择策略：**
- 长度差异最小化匹配
- 触发词精确匹配
- 支持自定义匹配逻辑

### 4. Pipeline - 主管道系统

整合所有组件的核心管道，实现端到端的对话处理。

**核心流程：**
1. **输入预处理** - 文本清洗和标准化
2. **语义分析** - 使用SentenceTransformer计算语义相似度
3. **触发器匹配** - 识别用户意图(带存储功能)和相关API以调用工具请求
4. **变量提取** - 使用滑动窗口技术提取关键信息
5. **并行处理** - 同时执行文档检索和API调用
6. **回复合成** - 整合所有信息生成智能回复

## 关键算法

### 1. 语义相似度计算

使用`thenlper/gte-large-zh`模型计算文本语义相似度：

```python
def extract_most_relevant_phrase(text: str, variable: str, model, window_size: int = 3, threshold=0.7):
    """使用滑动窗口从文本中提取与变量最相似的短语"""
    words = text.split()
    phrases = [' '.join(words[i:i + window_size]) for i in range(len(words) - window_size + 1)]
    
    phrase_embeddings = model.encode(phrases)
    var_embedding = model.encode([variable])[0]
    
    similarities = [cos_sim(var_embedding, phrase_embedding) for phrase_embedding in phrase_embeddings]
    max_idx = np.argmax(similarities)
    
    return phrases[max_idx] if similarities[max_idx] > threshold else ""
```

### 2. 触发器条件检查

多条件触发器系统，支持复杂的业务逻辑：

```python
# 检查所有变量是否都已填充
if all(var == 1 for var in info["bool"]):
    waitlist.append(trigger_key)
```

### 3. 异步任务管理

使用AsyncTaskManager实现并行处理：

```python
api_results, retriever_info = await self.task_manager.execute_tasks(api_list, user_input, param)
```

## 使用方式

### 基本初始化

```python
# 创建管道实例
folder_name = "data"  # 文档文件夹路径
prompt = "问题：{question}\n回答："
pipeline = Pipeline(folder_name, prompt)

# 初始化对话
pipeline.chat.add_human_message("你好!")
pipeline.chat.add_ai_message1("你好,有什么可以帮您!")
```

### 添加触发器

```python
# 添加API触发器
pipeline.add_trigger(
    trigger="天气预报",
    prompt_template=["绑定钥匙", "访问受限，未授权的请求"],
    sentence="希望用户补充如下可能的信息:\n{绑定钥匙}{访问受限，未授权的请求}",
    api="https://service.utcsoft.com/utcssc/api/ver/GLVER1000",
    tool=True
)
```

### 处理用户输入

```python
# 异步处理用户输入
user_input = "我现在遇到了一个问题：绑定钥匙时提示访问受限，未授权请求，这应该怎么办？"
trigger = ["天气预报"]  # 指定要检测的触发器
result = await pipeline.process_input(user_input, trigger)
```

## 技术栈

### 核心依赖
- **LangChain**: 文档处理和LLM集成框架
- **SentenceTransformers**: 语义相似度计算
- **HuggingFace Transformers**: 本地大语言模型
- **FAISS**: 高效向量相似度搜索
- **asyncio**: 异步任务处理

### 模型组件
- **嵌入模型**: `thenlper/gte-large-zh`（中文语义理解）
- **生成模型**: `Llama3-8B-instruct`（测试用，需改）
- **检索模型**: BM25 + FAISS混合检索


### 语义匹配配置
```python
similarity_threshold: float = 0.1   # 语义相似度阈值
window_size: int = 3               # 滑动窗口大小
phrase_threshold: float = 0.7      # 短语匹配阈值
```

## 性能特点

### 优势
- **多模态检索**: 结合关键词和语义检索，提高召回率
- **异步处理**: 并发执行多个任务，提升响应速度
- **智能缓存**: 避免重复计算，优化性能
- **模块化设计**: 各组件独立，易于扩展和维护

### 适用场景
- **智能客服系统**: 自动回答用户问题并调用相关API
- **知识问答助手**: 从文档库检索信息并生成答案
- **业务流程自动化**: 根据用户输入触发相应的业务逻辑
- **多轮对话系统**: 维护上下文进行连续对话

## 注意事项

1. **模型路径**: 确保指定正确的本地模型路径
2. **GPU支持**: 建议使用GPU加速推理过程
3. **内存管理**: 注意大模型的内存占用
4. **异步处理**: 正确处理异步任务的异常情况
5. **API限制**: 注意外部API的调用频率限制

## 总结

这个智能对话管道系统提供了一个完整的端到端解决方案，从用户输入到智能回复生成的全流程自动化处理。通过模块化设计和异步处理，系统既保证了功能的完整性，又具备了良好的扩展性和性能表现。