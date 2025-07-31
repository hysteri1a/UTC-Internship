import pandas as pd
import json

# 读取 Excel 文件（确保文件名和后缀正确）
df = pd.read_excel("base2.xlsx", header=None)

# 初始化结果列表
results = []

# 遍历每两列构成一个问题和答案行
for index, row in df.iterrows():
    problem = str(row[0]).strip() if pd.notna(row[0]) else ""
    answer = str(row[1]).strip() if pd.notna(row[1]) else ""

    if problem and answer:
        results.append({
            "problems": problem,
            "answer": answer
        })

# 保存为 JSON 文件
with open("./data/parsed_data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("转换完成，结果保存在 parsed_data.json")
