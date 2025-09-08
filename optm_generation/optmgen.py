import json
import re
import asyncio
import random
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# 初始化异步客户端
client = AsyncOpenAI(api_key="no-needed",
    base_url="http://localhost:4141/"
)

input_file = Path("match.jsonl")
summary_file = Path("sum.jsonl")
output_file = Path("optm.jsonl")


def extract_code_block(text: str) -> str:
    """
    提取最后一个被 ```json ... ``` 或 ``` ... ``` 包裹的部分。
    如果没有 code block，则返回原始文本。
    """
    pattern = re.compile(r"```(?:cpp|json)?\s*(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


async def process_line(idx: int, line: str, summary: str,
                       max_retries: int = 100, base_delay: float = 1.0) -> dict:
    """处理单行，带重试"""
    data = json.loads(line)
    summary_data = json.loads(summary)

    src_code = data.get("src_code", "")
    analysis_raw = data.get("analysis", "")
    analysis = extract_code_block(analysis_raw)

    # 从 summary 文件里取总结
    code_summary = summary_data.get("summary", "")

    # prompt = (
    #     "Here are list of optimization strategies:\n"
    #     f"{analysis}\n\n"
    #     "Here is a summary of the source code to help you understand it:\n"
    #     f"{code_summary}\n\n"
    #     "Now optimize the following C++ code by applying the above optimization strategies one by one. "
    #     "Output the current optimized code after each step, and finally output the complete optimized code using all optimizations.\n\n"
    #     # "After optimizing the C++ code, make sure all required headers are included and all macros are correctly defined.\n\n"
    #     f"{src_code}"
    #     "After optimizing the C++ code, add the header file `bits/stdc++.h` and make sure all macros are correctly defined.\n"
    #     ""
    # )

    prompt = (
        "Here are list of optimization strategies:\n"
        f"{analysis}\n\n"
        "Here is a summary of the source code to help you understand it:\n"
        f"{code_summary}\n\n"
        "Now optimize the following C++ code by applying the above optimization strategies one by one.\n"
        "\n"
        "For each step:\n"
        "1. State which optimization strategy is applied.\n"
        "2. Explain why this optimization improves performance.\n"
        "3. Explain why this modification does not change the logical behavior of the program "
        "(provide a brief reasoning or input/output comparison).\n"
        "4. Output the complete optimized C++ code after applying this step. Always output a full, "
        "independently compilable program.\n"
        "\n"
        "Important constraints:\n"
        "- The optimized code must preserve the original functionality and logic exactly.\n"
        "- If an optimization risks breaking correctness or changing behavior, skip it and explain why.\n"
        "- Always include required headers and ensure all macros are properly defined.\n"
        "- Each intermediate code must be syntactically correct and compilable.\n"
        "\n"
        "At the final step:\n"
        "- Combine all valid optimizations into one complete program.\n"
        "- Double-check and explicitly confirm that the final code is logically equivalent to the original.\n"
        "- Ensure the final code includes `#include <bits/stdc++.h>` and all macros are correctly defined.\n"
        "\n"
        "Here is the original source code:\n"
        f"{src_code}\n"
    )

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",  # 你要的模型
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            completion = response.choices[0].message.content
            completion_extracted = extract_code_block(completion)

            out_obj = dict(data)
            out_obj["__idx"] = idx
            out_obj["summary"] = code_summary
            out_obj["prompt"] = prompt
            out_obj["completion"] = completion
            out_obj["generated_answers"] = [completion_extracted]

            return out_obj

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                print(f"⚠️ Error on record {idx} attempt {attempt}/{max_retries}: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                print(f"❌ Record {idx} failed after {max_retries} attempts: {last_err}")
                out_obj = dict(data)
                out_obj["__idx"] = idx
                out_obj["summary"] = code_summary
                out_obj["prompt"] = prompt
                out_obj["completion"] = f"Error: {last_err}"
                out_obj["generated_answers"] = []
                return out_obj


async def main():
    # 读取输入
    with input_file.open("r", encoding="utf-8") as fin1, \
         summary_file.open("r", encoding="utf-8") as fin2:
        lines = fin1.readlines()
        summaries = fin2.readlines()

    if len(lines) != len(summaries):
        raise ValueError("match.jsonl 和 sum.jsonl 行数不一致，请检查。")

    tasks = [
        process_line(i, line, summary)
        for i, (line, summary) in enumerate(zip(lines, summaries))
    ]

    results = []
    with tqdm(total=len(tasks), desc="Processing") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)

    # 恢复原始行顺序
    results.sort(key=lambda r: r["__idx"])

    with output_file.open("w", encoding="utf-8") as fout:
        for r in results:
            r.pop("__idx", None)  # 移除内部索引
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ 已处理 {len(results)} 条，结果已保存到 {output_file}")


if __name__ == "__main__":
    asyncio.run(main())