import re
import json
import argparse
from pathlib import Path


def parse_args():
    """配置命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="对话日志解析工具 - 将客服对话记录转换为大模型训练格式"
    )

    # 必需参数
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入目录路径 (例如: ./data)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出文件路径 (例如: ./training_data.json)"
    )

    # 可选参数
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="启用详细输出模式"
    )

    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        choices=["utf-8", "gbk"],
        help="文件编码格式 (默认: utf-8)"
    )

    return parser.parse_args()


def parse_chat_log(content):
    # 分割不同日期的对话块
    date_blocks = re.split(r'-{20,}\s*日期：\d{4}/\d{2}/\d{2}\s*-{20,}', content)
    date_blocks = [b.strip() for b in date_blocks if b.strip()]

    result = []
    system_prompt = "我是一个客服, 我的职责是为客户解决各种难题。"

    for block in date_blocks:
        lines = block.split('\n')
        current_user = None
        current_role = None  # customer/service
        conversation = []
        pending_input = []
        pending_output = []

        for line in lines:
            # 检测消息行格式：用户名 (ID)        时间
            if re.match(r'^\s*[\S ]+\(\d+\)\s+\d{2}:\d{2}:\d{2}', line):
                # 提取消息信息
                parts = re.split(r'[()]', line)
                name = parts[0].strip()
                user_id = parts[1]
                time_part = re.search(r'\d{2}:\d{2}:\d{2}', line).group()
                content = line.split(time_part)[1].strip()

                # 判断角色
                is_service = "电子密钥客服" in name
                new_role = "service" if is_service else "customer"

                # 处理角色切换
                if current_role != new_role:
                    # 保存之前的对话
                    if current_role == "customer" and pending_input:
                        # 合并客户消息
                        full_input = "。".join(pending_input).replace("。。", "。")
                        pending_input = []

                        # 等待客服回复
                        if new_role == "service":
                            current_role = new_role
                            pending_output.append(content)
                        else:
                            # 没有客服回复的情况暂不处理
                            pass

                    elif current_role == "service" and pending_output:
                        # 合并客服回复
                        full_output = "。".join(pending_output).replace("。。", "。")
                        pending_output = []

                        # 添加到对话记录
                        if conversation:
                            conversation.append({
                                "input": full_input,
                                "output": full_output
                            })
                        else:
                            conversation.append({
                                "system": system_prompt,
                                "input": full_input,
                                "output": full_output
                            })

                        # 处理新消息
                        if new_role == "customer":
                            current_role = new_role
                            pending_input.append(content)
                        else:
                            # 连续客服消息合并
                            pending_output.append(content)
                    else:
                        # 初始化状态
                        current_role = new_role
                        if new_role == "customer":
                            pending_input.append(content)
                        else:
                            pending_output.append(content)

                else:
                    # 同一角色继续收集消息
                    if new_role == "customer":
                        pending_input.append(content)
                    else:
                        pending_output.append(content)

                # 过滤系统消息
                if any(msg in content for msg in ["客户已接入", "会话已"]):
                    if new_role == "customer":
                        pending_input.pop()
                    else:
                        pending_output.pop()

            # 处理消息内容中的媒体标记
            elif current_role == "customer" and pending_input:
                pending_input[-1] += " " + line.strip()
            elif current_role == "service" and pending_output:
                pending_output[-1] += " " + line.strip()

        # 处理最后一个对话对
        if pending_input and pending_output:
            full_input = "。".join(pending_input).replace("。。", "。")
            full_output = "。".join(pending_output).replace("。。", "。")
            if conversation:
                conversation.append({"input": full_input, "output": full_output})
            else:
                conversation.append({
                    "system": system_prompt,
                    "input": full_input,
                    "output": full_output
                })

        if conversation:
            result.append({"conversation": conversation})

    return result


def save_to_json(data, output_path, verbose=False):
    """保存JSON文件并添加状态反馈"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"✓ 成功保存 {len(data)} 条对话记录")
            print(f"↳ 输出路径: {output_path}")

    except IOError as e:
        print(f"× 文件保存失败: {str(e)}")
        raise


if __name__ == "__main__":
    import os
    args = parse_args()
    path1 = args.input
    print(path1)
    dirs = os.listdir(path1)
    print(dirs)
    all_data = []
    for num in dirs:
        input1 = path1+"/"+num
        input_path = Path(input1)
        try:
            if input_path.is_dir():
                txt_files = list(input_path.glob('*.txt'))
                if not txt_files:
                    raise FileNotFoundError(f"目录中没有找到txt文件: {args.input}")

                for txt_file in txt_files:
                    try:
                        with open(txt_file, 'r', encoding=args.encoding) as f:
                            raw_text = f.read()
                        file_data = parse_chat_log(raw_text)
                        all_data.extend(file_data)
                        if args.verbose:
                            print(f"处理文件: {txt_file.name} → 找到 {len(file_data)} 条对话")
                    except Exception as e:
                        print(f"处理文件 {txt_file} 时发生错误: {str(e)}")
            else:
                with open(input_path, 'r', encoding=args.encoding) as f:
                    raw_text = f.read()
                all_data = parse_chat_log(raw_text)
            print(len(all_data))
        except FileNotFoundError as e:
            print(f"错误：{str(e)}")
        except Exception as e:
            print(f"处理过程中发生错误：{str(e)}")
    save_to_json(all_data, args.output, args.verbose)