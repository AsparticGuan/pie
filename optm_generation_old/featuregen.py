import json
import asyncio
from string import Template
from tqdm import tqdm
from openai import AsyncOpenAI

# 初始化异步客户端
client = AsyncOpenAI()

# 读取 prompt 模板
with open("featureprompt.txt", "r", encoding="utf-8") as f:
    prompt_template = Template(f.read())

# 读取前 100 条数据
data = []
with open("test.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 100:
            data.append(json.loads(line))
        else:
            break

async def process_item(item):
    """异步处理单条数据"""
    src_code = item.get("src_code", "")
    tgt_code = item.get("tgt_code", "")
    prompt = prompt_template.safe_substitute(src_code=src_code, tgt_code=tgt_code)

    response = await client.chat.completions.create(
        # model="openai/gpt-4o-mini",
        model="google/gemini-2.5-pro",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for code optimization analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    analysis = response.choices[0].message.content.strip()

    output_item = dict(item)
    output_item["analysis"] = analysis
    return json.dumps(output_item, ensure_ascii=False)

async def main():
    # 创建任务
    tasks = [process_item(item) for item in data]

    # gather 保持顺序
    results = await asyncio.gather(*tasks)

    # tqdm 展示进度（不影响顺序）
    for _ in tqdm(range(len(results)), desc="Processing"):
        await asyncio.sleep(0)

    # 追加写入
    with open("feature.jsonl", "a", encoding="utf-8") as out_f:
        for line in results:
            if line:  # 过滤掉 None
                out_f.write(line + "\n")

    print("前 100 条已处理并保存到 feature.jsonl")

if __name__ == "__main__":
    asyncio.run(main())