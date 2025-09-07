import json

# 修改为你的 JSONL 文件路径
filepath = "/data/btguan/pie/test_sep7_2/addtl_stats.jsonl"

false_count = 0
true_but_low_speedup_count = 0

false_lines = []
true_low_speedup_lines = []

with open(filepath, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):  # 行号从 1 开始
        if i > 100:  # 只看前100行
            break
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # 遇到非法 JSON 行就跳过

        is_correct = obj.get("is_correct_best@1")
        speedup = obj.get("speedup_best@1")

        if is_correct is False:
            false_count += 1
            false_lines.append(i)
        elif is_correct is True and isinstance(speedup, (int, float)) and speedup < 1.1:
            true_but_low_speedup_count += 1
            true_low_speedup_lines.append(i)

print("is_correct_best@1: False 的个数 =", false_count)
print("对应的行号 =", false_lines)

print("is_correct_best@1: True 且 speedup_best@1 < 1.1 的个数 =", true_but_low_speedup_count)
print("对应的行号 =", true_low_speedup_lines)