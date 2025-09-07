import json
import re
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

client = AsyncOpenAI()

input_file = "feature.jsonl"
summary_file = "sum.jsonl"
output_file = "optm.jsonl"

def extract_code_block(text: str) -> str:
    """
    提取最后一个被 ```json ... ``` 或 ``` ... ``` 包裹的部分。
    如果没有 code block，则返回原始文本。
    """
    pattern = re.compile(r"```(?:cpp|json)?\s*(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()  # 只取最后一个
    return text.strip()

async def process_line(line: str, summary: str) -> dict:
    data = json.loads(line)
    summary_data = json.loads(summary)

    src_code = data.get("src_code", "")
    analysis_raw = data.get("analysis", "")
    analysis = extract_code_block(analysis_raw)

    # 从 summary 文件里取总结
    code_summary = summary_data.get("summary", "")

    prompt = (
        "Here are list of optimization strategies:\n"
        f"{analysis}\n\n"
        "Here is a summary of the source code to help you understand it:\n"
        f"{code_summary}\n\n"
        "Now optimize the following C++ code by applying the above optimization strategies one by one. "
        "After optimizing the C++ code, make sure all required headers are included and all macros are correctly defined.\n\n"
        f"{src_code}"
    )

    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",  # 可替换成你需要的模型
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    completion = response.choices[0].message.content
    completion_extracted = extract_code_block(completion)

    out_obj = dict(data)  # 复制原始字段
    out_obj["summary"] = code_summary
    out_obj["prompt"] = prompt
    out_obj["completion"] = completion
    out_obj["generated_answers"] = [completion_extracted]

    return out_obj

async def main():
    with open(input_file, "r", encoding="utf-8") as fin1, \
         open(summary_file, "r", encoding="utf-8") as fin2:
        lines = fin1.readlines()
        summaries = fin2.readlines()

    if len(lines) != len(summaries):
        raise ValueError("feature.jsonl 和 sum.jsonl 行数不一致，请检查。")

    results = []
    for line, summary in tqdm_asyncio(zip(lines, summaries), desc="Processing", total=len(lines)):
        result = await process_line(line, summary)
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    asyncio.run(main())