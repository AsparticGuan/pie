import json
import asyncio
import random
from string import Template
from tqdm import tqdm
from openai import AsyncOpenAI

# 初始化异步客户端
client = AsyncOpenAI(
    api_key="no-needed",
    base_url="http://localhost:4141/"
)

# 读取 prompt 模板
with open("featureprompt.txt", "r", encoding="utf-8") as f:
    prompt_template = Template(f.read())

# 读取train/val集数据
data = []
with open("val.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # if i < 100:
        data.append(json.loads(line))
        # else:
        #     break

async def call_with_retry(prompt, max_retries=100, base_delay=1.0):
    """带重试的 API 调用"""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for code optimization analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response
        except Exception as e:
            wait = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"⚠️ 调用失败: {e}, {wait:.1f}s 后重试 (第 {attempt+1}/{max_retries} 次)")
            await asyncio.sleep(wait)
    print("❌ 超过最大重试次数，跳过该条数据")
    return None

async def process_item(item):
    """异步处理单条数据"""
    src_code = item.get("src_code", "")
    tgt_code = item.get("tgt_code", "")
    prompt = prompt_template.safe_substitute(src_code=src_code, tgt_code=tgt_code)

    response = await call_with_retry(prompt)
    if not response:
        return None

    analysis = response.choices[0].message.content.strip()
    output_item = dict(item)
    output_item["analysis"] = analysis
    return json.dumps(output_item, ensure_ascii=False)

async def main():
    # 创建任务
    tasks = [process_item(item) for item in data]

    # gather 保持顺序
    results = await asyncio.gather(*tasks)

    # tqdm 展示进度（这里用异步模拟进度条）
    for _ in tqdm(range(len(results)), desc="Processing"):
        await asyncio.sleep(0)

    with open("feature.jsonl", "w", encoding="utf-8") as out_f:
        for line in results:
            if line:  # 过滤掉 None
                out_f.write(line + "\n")

    print("已处理并保存到 feature.jsonl")

if __name__ == "__main__":
    asyncio.run(main())