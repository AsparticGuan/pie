import asyncio
import json
import random
import time
from pathlib import Path
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# Initialize OpenAI async client
client = AsyncOpenAI(
    api_key="no-needed",
    base_url="http://localhost:4141/"
)

async def summarize_code(record: dict, max_retries: int = 100, base_delay: float = 1.0):
    """Call OpenAI API to summarize source code with retries"""
    snippet = record.get("src_code", "")
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarize the purpose of the following code and place the summary in a txt code block without outputting other content.\n{snippet}"
                    }
                ]
            )
            summary = response.choices[0].message.content.strip()
            record["summary"] = summary
            return record

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                # 指数退避 + 随机抖动
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                print(f"⚠️ Error on record {record.get('__idx')} (attempt {attempt}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                record["summary"] = f"Error after {max_retries} attempts: {last_err}"
                print(record["summary"])
                return record

async def main():
    input_file = Path("test.jsonl")
    output_file = Path("sum.jsonl")

    # load first 100 records, with index
    records = []
    with input_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            record = json.loads(line)
            record["__idx"] = i
            records.append(record)

    tasks = [summarize_code(record) for record in records]

    results = []
    # 边完成边收集 + 更新进度条
    with tqdm(total=len(tasks), desc="Summarizing") as pbar:
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            pbar.update(1)

    # 最后恢复顺序
    results.sort(key=lambda r: r["__idx"])

    with output_file.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Processed {len(results)} records, saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())