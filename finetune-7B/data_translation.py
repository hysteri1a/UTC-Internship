import json

# 从 data1.json 文件读取原始数据集
with open('augmented_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理数据
output_data = []

for entry in data:
    messages = []
    for idx, conv in enumerate(entry["conversation"]):
        if idx == 0:
            # 第一个对话作为 system
            messages.append({
                "role": "system",
                "content": "电子签章智能助手"  # 固定的system内容
            })

        # 添加用户消息
        messages.append({
            "role": "user",
            "content": conv["input"].strip()
        })

        # 添加助手回复
        messages.append({
            "role": "assistant",
            "content": conv["output"].strip()
        })

    # 将处理后的数据添加到最终的输出数据中
    output_data.append({"messages": messages})

# 保存处理后的数据到 data2.json 文件
with open('augmented-standard-data.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("数据已成功转换并保存到 data2.json")
