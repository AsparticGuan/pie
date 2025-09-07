import json
import asyncio
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio

# Initialize async client (make sure OPENAI_API_KEY is set in the environment)
client = AsyncOpenAI()

input_file = "k4prompt.jsonl"
output_file = "try.jsonl"


def build_prompt(user_input: str) -> list:
    """Build prompt with a short instruction before the few-shot examples in user_input"""
    user_message = (
        "You are given several pairs of 'slower version' and 'optimized version' C/C++ code snippets. "
        "Carefully learn the optimization patterns from the given examples, and then generate the optimized "
        "version for the final slower code at the end.\n\n"
        f"{user_input}"
    )

    return [
        {
            "role": "system",
            "content": "You are an expert C/C++ assistant that generates optimized code from slower code versions."
        },
        {"role": "user", "content": user_message}
    ]


# Retry decorator
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
async def generate_answer(user_input: str) -> str:
    """Call OpenAI API with retry logic"""
    messages = build_prompt(user_input)
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()


async def process_line(line: str) -> str:
    """Process one line of JSONL: call API and return new JSON string"""
    data = json.loads(line)
    user_input = data.get("input", "")
    try:
        answer = await generate_answer(user_input)
        data["generated_answers"] = answer
    except Exception as e:
        data["generated_answers"] = f"ERROR: {e}"
    return json.dumps(data, ensure_ascii=False)


async def main():
    # Read input file
    with open(input_file, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    # Concurrency limit
    semaphore = asyncio.Semaphore(5)

    async def sem_task(line):
        async with semaphore:
            return await process_line(line)

    # Run with progress bar
    results = await tqdm_asyncio.gather(
        *(sem_task(line) for line in lines),
        desc="Processing",
        total=len(lines)
    )

    # Write results
    with open(output_file, "w", encoding="utf-8") as fout:
        for res in results:
            fout.write(res + "\n")

    print(f"✅ Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())