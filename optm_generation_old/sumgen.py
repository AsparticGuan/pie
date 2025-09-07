import asyncio
import json
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI

# Initialize OpenAI async client
client = AsyncOpenAI()

async def summarize_code(record: dict):
    """Call OpenAI API to summarize source code"""
    try:
        snippet = record.get("src_code", "")
        response = await client.chat.completions.create(
            model="google/gemini-2.5-pro",   # choose appropriate model
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize the purpose of the following code and place the summary in a txt code block without outputting other content.\n{snippet}"
                }
            ],
        )
        summary = response.choices[0].message.content.strip()
        record["summary"] = summary
        return record
    except Exception as e:
        record["summary"] = f"Error: {e}"
        return record

async def main():
    input_file = Path("test.jsonl")
    output_file = Path("sum.jsonl")

    # Load first 100 records
    records = []
    with input_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            records.append(json.loads(line))

    # Create coroutine tasks
    tasks = [summarize_code(record) for record in records]

    # Run tasks concurrently, ordered
    results = await asyncio.gather(*tasks)

    # tqdm 展示进度（这里是完成之后展示 100%）
    for _ in tqdm(range(len(results)), desc="Summarizing"):
        await asyncio.sleep(0)

    # Write results (keep input order)
    with output_file.open("a", encoding="utf-8") as f:
        for record in results:
            if record:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n✅ Processed {len(results)} records, saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())