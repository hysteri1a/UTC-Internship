# main.py
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.graph.message import MessageGraph
from tools import multimodal_tool, word_tool_wrapper
from tools import multi_tool, word_tool   # 这是个 Tool 实例

chat_model = ChatOpenAI(
    base_url="http://192.168.1.224:8000/v1",   # 指向你的服务地址
    api_key="EMPTY",                           # 如果没认证，也可传空
    model="qwen3-32b",                         # 模型名必须与你服务端匹配
).bind_tools([multi_tool,word_tool])

# 2) 注册唯一工具
tool_node = ToolNode([multimodal_tool,word_tool])

# 3) 构建 LangGraph
graph = MessageGraph()
graph.add_node("chat", chat_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("chat")
graph.add_edge("chat", "tools")
graph.set_finish_point("tools")
app = graph.compile()

# 运行一次
initial_messages = [
    HumanMessage(content=(
        "请调用 multimodal_inference(image_path=\"./temp/test18.png\", "
        "audio_path=\" \", prompt=\"请识别图片,告诉我这个图片是干什么的。\")函数。"
        " "
    ))
]
final_state = app.invoke(initial_messages)

# 提供初始消息
initial = [HumanMessage(content='请调用 word_tool(file_path="./test.docx", prompt="请总结文档的主要内容并提取图表信息")')]

# 执行
final = app.invoke(initial)

# 打印输出
for msg in final:
    if isinstance(msg, AIMessage):
        print("[AI] ", msg.content.strip())
    elif isinstance(msg, ToolMessage):
        print("[Tool Output]", msg.content.strip())

# 5) 打印
for msg in final_state:
    if isinstance(msg, AIMessage):
        print("[AI]", msg.content.strip())
    elif isinstance(msg, ToolMessage):
        # 只打印工具输出内容
        print("[Tool Output]", msg.content.strip())


