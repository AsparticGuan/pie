import json
import re

input_file = "k4.jsonl"
output_file = "k4_out.jsonl"

code_pattern = re.compile(r"```cpp\s*(.*?)\s*```", re.DOTALL)

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)

        ans = obj.get("generated_answers")
        if isinstance(ans, str):
            # 查找所有cpp代码块
            code_blocks = code_pattern.findall(ans)
            if code_blocks:
                obj["generated_answers"] = [block.strip() for block in code_blocks]
            else:
                obj["generated_answers"] = [ans.strip()]

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"处理完成，结果保存在 {output_file}")