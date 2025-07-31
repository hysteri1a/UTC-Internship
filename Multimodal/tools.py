# tools.py
from langchain_core.tools import tool, Tool
from pydantic import BaseModel, Field


class MultiInput(BaseModel):
    image_path: str = Field(..., description="本地图片文件路径或 URL")
    audio_path: str = Field(..., description="本地音频文件路径或 URL")
    prompt: str = Field(..., description="对图音信息的文字提示，如“请描述...”")

@tool(
    "multimodal_inference",
    description="同时处理图片和音频，并根据提示生成综合回答",
    args_schema=MultiInput
)
def multimodal_tool(image_path: str, audio_path: str, prompt: str) -> str:
    from combined_tool import multimodal_inference
    return multimodal_inference(image_path, audio_path, prompt)

multi_tool = Tool(
    name="multimodal_inference",
    func=multimodal_tool,
    description="一次性处理图片和音频，返回综合回答",
    args_schema=MultiInput
)




# 输入 schema
class WordToolInput(BaseModel):
    file_path: str = Field(..., description="Word 文件路径")
    prompt: str = Field(..., description="提取文档内容的提示词")

@tool(
    "word_tool",
    description="读取 Word 文档，并结合提示词进行总结与图表提取",
    args_schema=WordToolInput
)
def word_tool(file_path: str, prompt: str) -> str:
    from combined_tool import extract_from_word
    return extract_from_word(file_path, prompt)

# 绑定为 LangGraph Tool
word_tool_wrapper = Tool(
    name="word_tool",
    func=word_tool,
    description="用于分析 Word 文档内容和图表",
    args_schema=WordToolInput
)