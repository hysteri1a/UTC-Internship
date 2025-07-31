import re, os, asyncio,json, aiofiles, sys
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from typing import Dict, Any
from fastapi import UploadFile, File, HTTPException, Form
from typing import Annotated
from pathlib import Path

def normalize(s: str) -> str:
    """
    去掉“第X章”、所有非字母数字汉字字符，转小写。
    """
    # 去掉“第X章”这样的前缀
    s = re.sub(r'第[一二三四五六七八九十]+章', '', s)
    # 去掉所有非字母数字汉字
    s = re.sub(r'\W+', '', s)
    return s.lower()

def lcs_length(a: str, b: str) -> int:
    """
    计算 a 和 b 的最长公共子序列长度。
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

def lcs_similarity(a: str, b: str) -> float:
    """
    基于 LCS 长度计算相似度：lcs_len / max(len(a), len(b))
    """
    if not a or not b:
        return 0.0
    lcs = lcs_length(a, b)
    return lcs / max(len(a), len(b))

def safe_eval(expr: str) -> float:
    """安全地计算数学表达式"""
    import re
    import ast
    import math
    try:
        # 只允许数学表达式中的安全字符和函数
        allowed_names = {
            k: v for k, v in math.__dict__.items()
            if not k.startswith("_")
        }
        allowed_names.update({"abs": abs, "round": round})

        # 编译表达式并检查节点
        code = compile(expr, "<string>", "eval")
        for node in ast.walk(ast.parse(expr)):
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                raise NameError(f"Use of {node.id} is not allowed")

        # 执行计算
        return eval(code, {"__builtins__": {}}, allowed_names)

    except:
        return 0.0

def iter_block_items(parent):
    """
    Yield each paragraph and table child from parent, in document order.
    `parent` can be a Document or _Cell.
    """
    for child in parent.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def _copy_run_format(src_run, tgt_run):
    """
    把 src_run（原文档中的 run）里所有跟字体／样式有关的
    属性，尽量全都搬到 tgt_run（新文档里的 run）上。
    """
    # 1. 直接拷贝粗体、斜体、下划线
    tgt_run.bold = src_run.bold
    tgt_run.italic = src_run.italic
    tgt_run.underline = src_run.underline

    # 2. 拷贝字体大小
    if src_run.font.size:
        tgt_run.font.size = src_run.font.size

    # 3. 拷贝字体名称（西文）
    if src_run.font.name:
        tgt_run.font.name = src_run.font.name

    # 4. 显式设置东亚（中文）字体，以及 hAnsi（西文）和 ascii
    #    这样 Word 打开时就会用同一个字体来渲染中／西文
    rPr = tgt_run._r.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    font_name = src_run.font.name or "宋体"  # 如果原 run 真的拿不到名字，可以给个默认
    # 设置三大字体属性
    rFonts.set(qn('w:eastAsia'), font_name)
    rFonts.set(qn('w:ascii'), font_name)
    rFonts.set(qn('w:hAnsi'), font_name)

BIDDING_KEYS = ["bidding_announcement", "technical_specification", "price_requirement"]
RESPONSE_KEYS = ["business_response", "technical_response", "price_response"]
ALL_KEYS = BIDDING_KEYS + RESPONSE_KEYS
sessions: Dict[str, Dict[str, Any]] = {}
# Helpers
def validate_session_exists(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

def validate_file_type(file_type: str) -> None:

    if file_type not in ALL_KEYS:
        raise HTTPException(status_code=400, detail="Invalid file_type")

# Dependency for upload parameters
async def get_upload_params(
        session_id: Annotated[str, Form()],
        file_type: Annotated[str, Form()],
        file: Annotated[UploadFile, File()]
) -> Dict[str, Any]:
    return {"session_id": session_id, "file_type": file_type, "file": file}

import asyncio
async def json_to_readable(part_data, delay=0.05):
    """流式输出评分表可读格式的生成器函数 - 优化版"""

    scoring_tables = part_data.get("scoring_tables", [])
    if not scoring_tables:
        yield "📋 暂无评分表信息\n"
        return

    # 遍历所有评分表
    for table_idx, table in enumerate(scoring_tables, 1):
        # 表头信息 - 使用卡片样式
        yield f"\n🏆 评分表 {table_idx}\n"
        yield f"{'=' * 60}\n"
        await asyncio.sleep(delay)

        # 基本信息
        table_name = table.get('table_name', '未命名评分表')
        table_desc = table.get('table_description', '暂无描述')
        total_range = format_score_range(table.get('total_score_range', {}))
        weight = table.get('weight', '未设置')

        yield f" 名称：{table_name}\n"
        await asyncio.sleep(delay)

        yield f" 描述：{table_desc}\n"
        await asyncio.sleep(delay)

        yield f" 总分范围：{total_range}\n"
        await asyncio.sleep(delay)

        yield f"  权重：{weight}\n"
        await asyncio.sleep(delay)

        yield f"\n{'─' * 60}\n"
        await asyncio.sleep(delay)

        # 遍历所有节
        sections = table.get("sections", [])
        for section_idx, section in enumerate(sections, 1):
            section_name = section.get('section_name', f'第{section_idx}节')
            section_range = format_score_range(section.get('section_score_range', {}))

            yield f"\n {section_name}\n"
            yield f"   分数范围：{section_range}\n"
            await asyncio.sleep(delay)

            # 遍历所有小节
            subsections = section.get("subsections", [])
            for subsection_idx, subsection in enumerate(subsections, 1):
                subsection_name = subsection.get('subsection_name', f'小节{subsection_idx}')
                subsection_range = format_score_range(subsection.get('subsection_score_range', {}))

                yield f"\n    {subsection_name}\n"
                yield f"      分数范围：{subsection_range}\n"
                await asyncio.sleep(delay)

                # 遍历所有评分项
                items = subsection.get("items", [])
                if not items:
                    yield f"        暂无评分项\n"
                    await asyncio.sleep(delay)
                    continue

                yield f"\n       评分项目：\n"
                await asyncio.sleep(delay)

                for item_idx, item in enumerate(items, 1):
                    item_name = item.get('item_name', f'项目{item_idx}')
                    description = item.get('description', '暂无描述')
                    score_range = format_score_range(item.get('score_range', {}))
                    scoring_method = item.get('scoring_method', '暂无评分方法')

                    # 使用更清晰的格式
                    yield f"      {item_idx}.  {item_name}\n"
                    await asyncio.sleep(delay)

                    yield f"          描述：{description}\n"
                    await asyncio.sleep(delay)

                    yield f"          分数：{score_range}\n"
                    await asyncio.sleep(delay)

                    yield f"          评分方法：{scoring_method}\n"
                    await asyncio.sleep(delay)

                    if item_idx < len(items):
                        yield f"         {'-' * 40}\n"
                        await asyncio.sleep(delay)

        # 表结束
        yield f"\n{'=' * 60}\n"
        if table_idx < len(scoring_tables):
            yield f"{'⬇' * 20}\n"
        await asyncio.sleep(delay)

# 方法2: 使用JSON但在接收端正确处理
async def async_line_chunks(data):
    import json
    """使用JSON但确保换行符被正确处理"""
    data_str = str(data) if not isinstance(data, str) else data
    lines = data_str.split('\n')

    for line in lines:
        # 将每行包装在JSON中
        json_data = json.dumps({
            'status': 'process',
            'data': line + '\n',  # 明确添加换行符
            'message': ''
        }, ensure_ascii=False)  # 确保中文字符不被转义
        yield json_data
        await asyncio.sleep(0.05)

def format_score_range(score_range):
    """格式化分数范围显示"""
    if isinstance(score_range, list) and len(score_range) == 2:
        return f"{score_range[0]} ~ {score_range[1]}分"
    return str(score_range)

async def execute_image_recognition_for_file(file_path, output_folder, file_name):
    """为单个文件执行图片识别"""
    try:
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 更新配置文件
        update_image_config_for_file(file_path, output_folder)

        # 执行图片识别脚本
        script_path = "./image_algorithm/main.py"

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"图片识别脚本不存在: {script_path}")

        # 异步执行脚本
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd()
        )

        stdout, stderr = await process.communicate()
        print(stderr)
        if process.returncode != 0:
            error_msg = stderr.decode('GBK') if stderr else "未知错误"
            raise RuntimeError(f"文件 {file_name} 图片识别脚本执行失败: {error_msg}")

        return stdout.decode('GBK') if stdout else ""

    except Exception as e:
        print(f"处理文件 {file_name} 时发生错误: {e}")
        raise


def update_image_config_for_file(file_path, output_folder):
    """为单个文件更新 image_algorithm/config.py 中的路径配置"""
    config_path = Path("./image_algorithm/config.py")

    # 读取现有配置
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式替换路径
    import re
    file_path = file_path.replace("\\", "\\\\")
    # 替换INPUT_PATH
    input_pattern = r'INPUT_PATH = r"[^"]*"'
    input_replacement = f'INPUT_PATH = r"{file_path}"'
    print(input_pattern)
    print(input_replacement)
    content = re.sub(input_pattern, input_replacement, content)

    # 替换OUTPUT_FOLDER
    output_pattern = r'OUTPUT_FOLDER = "[^"]*"'
    output_replacement = f'OUTPUT_FOLDER = "{output_folder}"'
    content = re.sub(output_pattern, output_replacement, content)

    # 替换COMPRESSED_FOLDER
    compressed_folder = os.path.join(output_folder, "compressed")
    compressed_pattern = r'COMPRESSED_FOLDER = os\.path\.join\(OUTPUT_FOLDER, "[^"]*"\)'
    compressed_replacement = f'COMPRESSED_FOLDER = os.path.join(OUTPUT_FOLDER, "compressed")'
    content = re.sub(compressed_pattern, compressed_replacement, content)

    # 写回配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)


async def read_image_analysis_result(output_folder):
    """读取指定输出文件夹中的图片分析结果文件"""
    result_file_path = os.path.join(output_folder, "image_analysis_results.json")

    if not os.path.exists(result_file_path):
        return None

    try:
        async with aiofiles.open(result_file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return json.loads(content)
    except Exception as e:
        print(f"读取图片分析结果失败 ({result_file_path}): {e}")
        return None



# 辅助函数：批量处理配置更新（如果需要同时处理多个文件）
async def batch_update_configs_and_execute(files_info):
    """批量更新配置并执行图片识别"""
    tasks = []

    for key, info in files_info.items():
        if info:
            file_path = info["path"]
            file_name = Path(file_path).stem
            output_folder = f"./image_algorithm/output/{file_name}"

            task = asyncio.create_task(
                execute_image_recognition_for_file(file_path, output_folder, file_name)
            )
            tasks.append({
                'task': task,
                'file_key': key,
                'file_name': file_name,
                'output_folder': output_folder
            })

    return tasks