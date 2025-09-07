import json

def count_printf_in_analysis(jsonl_path, max_lines=100):
    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            try:
                obj = json.loads(line.strip())
                analysis = obj.get("prompt", "")
                if "printf" in analysis:
                    count += 1
            except json.JSONDecodeError:
                print(f"❌ 第 {i+1} 行解析失败，跳过。")
    return count

# 使用示例
jsonl_file = "/data/btguan/pie/test_sep6/addtl_stats.jsonl"
result = count_printf_in_analysis(jsonl_file)
print("前100行analysis字段包含`printf`的条数:", result)