import json
import glob

system_message = "现在你是一名专业客服，当对方和你闲聊时，可以适当聊天以改善关系。"


def process_dialogue(dialogue):
    # 去除句子中的空格并清理格式
    cleaned = [s.replace(' ', '') for s in dialogue]

    # 创建对话轮次列表
    rounds = []
    for i in range(0, len(cleaned) - 1, 2):
        input_sentence = cleaned[i]
        output_sentence = cleaned[i + 1]
        rounds.append((input_sentence, output_sentence))

    if not rounds:
        return None

    # 构建对话结构
    conversation = []
    total_rounds = len(rounds)

    for idx, (input_sen, output_sen) in enumerate(rounds):
        is_last = idx == total_rounds - 1
        input_str = f"{input_sen}\n"
        output_str = f"{output_sen}" + ("" if is_last else "\n\n")

        if idx == 0:
            conversation.append({
                "system": system_message,
                "input": input_str,
                "output": output_str
            })
        else:
            conversation.append({
                "input": input_str,
                "output": output_str
            })

    return conversation


def main():
    output_data = []

    # 读取所有jsonl文件
    for file_path in glob.glob("./daily-data/*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    dialogue = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # 处理对话并添加到结果
                processed = process_dialogue(dialogue)
                if processed:
                    output_data.append({"conversation": processed})

    # 保存结果
    with open("daily-data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4, separators=(",", ": "))


if __name__ == "__main__":
    main()