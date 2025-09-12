import asyncio
import json
import random
from pathlib import Path
from string import Template

from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# 初始化异步 OpenAI 客户端
client = AsyncOpenAI(
    api_key="no-needed",        # 换成真实 key 时使用
    base_url="http://localhost:4141/"
)

async def process_program(record: dict, prompt_template: Template,
                          max_retries: int = 100, base_delay: float = 1.0) -> dict:
    """调用大模型，传入代码，返回优化特征，带错误重试"""
    src_code = record.get("src_code", "")
    last_err = None

    if not src_code:
        record["optimized_features"] = ""
        return record

    prompt = prompt_template.safe_substitute(program=src_code)

    for attempt in range(1, max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1",
                messages=[{
                    "role": "system", "content": "You are an expert C/C++ assistant that extracts optimization features from unoptimized code versions.",
                    "role": "user", "content": prompt}
                ],
            )
            optimized_features = response.choices[0].message.content.strip()
            record["optimized_features"] = optimized_features
            return record

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                print(f"⚠️ Error on record {record['__idx']} "
                      f"(attempt {attempt}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                record["optimized_features"] = f"Error after {max_retries} attempts: {last_err}"
                return record


def load_prompt_template(file_path: str) -> Template:
    """从 txt 文件中加载 Prompt 模板"""
    with open(file_path, "r", encoding="utf-8") as f:
        return Template(f.read())


async def main():
    input_file = Path("test.jsonl")
    output_file = Path("extract_feature.jsonl")
    prompt_file = Path("extract_feature_prompt.txt")

    prompt_template = load_prompt_template(prompt_file)

    # 读取前 100 条数据，添加索引
    records = []
    with input_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            try:
                data = json.loads(line)
                data["__idx"] = i
                records.append(data)
            except Exception as e:
                print(f"⚠️ Error parsing JSON line {i}: {e}")

    # 并发任务
    tasks = [process_program(r, prompt_template) for r in records]

    results = []
    # tqdm 异步进度条
    with tqdm(total=len(tasks), desc="Processing") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)

    # 恢复原始顺序
    results.sort(key=lambda r: r["__idx"])

    # 写入输出文件
    with output_file.open("w", encoding="utf-8") as f:
        for r in results:
            # 去掉内部索引
            r.pop("__idx", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ 已处理 {len(results)} 条，结果保存在 {output_file}")


if __name__ == "__main__":
    asyncio.run(main())