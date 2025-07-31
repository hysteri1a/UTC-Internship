#add_tool.py

import argparse
from pathlib import Path

TEMPLATE = '''
# —— 定义 {class_name} 输入 schema ——
class {class_name}(BaseModel):
    path: str = Field(
        ...,
        description="{input_desc}"
    )

@tool(
    "{tool_name}",
    description="{tool_desc}",
    args_schema={class_name}
)
def {tool_name}(path: str) -> str:
    """
    {tool_desc}

    参数:
    - path (str): 本地路径或 URL

    返回:
    - str: 工具 `{tool_name}` 的返回结果
    """
    from {module_name} import {call_fn}
    return {call_fn}(path)
'''

def snake_to_camel(name):
    return ''.join(word.capitalize() for word in name.split('_'))

def main():
    parser = argparse.ArgumentParser(description="自动添加工具到 tools.py")
    parser.add_argument("--tool-name", required=True, help="工具函数名（如 image_tool）")
    parser.add_argument("--tool-desc", required=True, help="@tool 描述")
    parser.add_argument("--input-desc", required=True, help="输入 path 字段的描述")
    parser.add_argument("--module-name", default="image_model", help="调用的模块名（默认 image_model）")
    parser.add_argument("--call-fn", default="predict_image_caption", help="调用的函数名（默认 predict_image_caption）")
    parser.add_argument("--tools-file", default="tools.py", help="目标 tools.py 路径")

    args = parser.parse_args()

    class_name = snake_to_camel(args.tool_name)

    content = TEMPLATE.format(
        class_name=class_name,
        tool_name=args.tool_name,
        tool_desc=args.tool_desc,
        input_desc=args.input_desc,
        module_name=args.module_name,
        call_fn=args.call_fn
    )

    tools_path = Path(args.tools_file)
    with tools_path.open("a", encoding="utf-8") as f:
        f.write("\n" + content)

    print(f"✅ 工具 `{args.tool_name}` 已添加到 {tools_path.absolute()}")

if __name__ == "__main__":
    main()
