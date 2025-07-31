import re
import pandas as pd
import json

def preprocess_text(text):
    # 定义分隔符正则表达式（分号和空格为顶级分隔符）
    top_level_delimiters = r'[;\s]+'
    clauses = re.split(top_level_delimiters, text)

    result_pairs = []

    for clause in clauses:
        if not clause.strip():
            continue

        # 处理引号内容
        quote_match = re.search(r'[“"](.*?)[”"]', clause)
        if quote_match:
            start, end = quote_match.span()
            state = clause[:start].strip()
            result = quote_match.group(1).strip()
            if state:
                result_pairs.append({"state": state, "result": result})
            continue

        # 处理冒号
        if ':' in clause or '：' in clause:
            state, result = re.split(r'[:：]', clause, 1)
            state = state.strip()
            result = result.strip()
            result_pairs.append({"state": state, "result": result})
            continue

        # 处理逗号
        if ',' in clause:
            # 递归处理链式补充关系
            parts = [p.strip() for p in clause.split(',') if p.strip()]
            for i in range(len(parts) - 1):
                result_pairs.append({
                    "state": parts[i],
                    "result": parts[i + 1]
                })
            continue

    return result_pairs


# 读取文件
file_path = "脱敏.xlsx"
data = pd.read_excel(file_path)

# 初始化结果存储
processed_data = []

# 对每一行进行处理
for index, row in data.iterrows():
    question = row["问题"]
    answer = row["答案"]

    # 对问题进行预处理
    question_pairs = preprocess_text(question)

    # 处理空 processed 的情况
    if not question_pairs:  # 如果预处理结果为空
        question_pairs = [{"state": question, "result": ""}]  # 使用原问题作为 state，result 为空

    # 将结果保存
    processed_data.append({
        "problems": question,
        "processed": question_pairs,
        "answer": answer,
    })

# 将结果保存到 data.json 文件
with open("rag-data.json", "w", encoding="utf-8") as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)

print("数据已保存到 rag-data.json 文件")

# 将 processed_data 转换为 DataFrame
output_data = []
for item in processed_data:
    for pair in item["processed"]:
        output_data.append({
            "state": pair["state"],
            "result": pair["result"],
            "answer": item["answer"]
        })

# 创建 DataFrame
output_df = pd.DataFrame(output_data)

# 保存到 Excel 文件
output_df.to_excel("processed_data.xlsx", index=False, columns=["state", "result", "answer"])

print("数据已保存到 processed_data.xlsx 文件")