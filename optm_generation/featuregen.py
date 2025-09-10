import json
import asyncio
import random
from pathlib import Path
from string import Template

from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# 初始化异步客户端
client = AsyncOpenAI(
    api_key="no-needed",   # 换成真实 key 时替换
    base_url="http://localhost:4141/"
)


def load_prompt_template(file_path: str) -> Template:
    """从文本文件加载 Prompt 模板"""
    with open(file_path, "r", encoding="utf-8") as f:
        return Template(f.read())


async def process_item(record: dict, prompt_template: Template,
                       semaphore: asyncio.Semaphore,
                       max_retries: int = 100, base_delay: float = 1.0) -> dict:
    """处理单条数据，带重试和并发限制"""
    src_code = record.get("src_code", "")
    tgt_code = record.get("tgt_code", "")
    last_err = None

    if not src_code and not tgt_code:
        record["analysis"] = ""
        return record

    prompt = prompt_template.safe_substitute(src_code=src_code, tgt_code=tgt_code)

    async with semaphore:  # 限制并发量
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for code optimization analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                analysis = response.choices[0].message.content.strip()
                record["analysis"] = analysis
                return record

            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    print(f"⚠️ Error on record {record['__idx']} "
                          f"(attempt {attempt}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    record["analysis"] = f"Error after {max_retries} attempts: {last_err}"
                    return record


async def main():
    input_file = Path("val.jsonl")
    output_file = Path("feature.jsonl")
    prompt_file = Path("featureprompt.txt")

    prompt_template = load_prompt_template(prompt_file)

    # 并发限制，比如同时最多 100 个任务
    max_concurrency = 100
    semaphore = asyncio.Semaphore(max_concurrency)

    # 加载数据
    records = []
    with input_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                data["__idx"] = i
                records.append(data)
            except Exception as e:
                print(f"⚠️ Error parsing JSON line {i}: {e}")

    tasks = [process_item(r, prompt_template, semaphore) for r in records]

    results = []
    with tqdm(total=len(tasks), desc="Processing") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)

    # 恢复顺序
    results.sort(key=lambda r: r["__idx"])

    # 写入输出文件
    with output_file.open("w", encoding="utf-8") as out_f:
        for r in results:
            r.pop("__idx", None)  # 删除内部索引
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ 已处理 {len(results)} 条，结果保存在 {output_file}")


if __name__ == "__main__":
    asyncio.run(main())