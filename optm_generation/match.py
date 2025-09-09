import json
import jsonlines
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# ---------------- 工具函数 -------------------

def fix_and_parse_json(bad_json_str, line_no=None):
    """修复并解析非法 JSON"""
    def attempt_parse(s, stage):
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            print(f"[WARN] Line {line_no}: JSON decode failed in {stage}: {e}")
            snippet = s[max(0, e.pos-50):e.pos+50]
            print("Problematic snippet:")
            print(snippet)
            return None

    data = attempt_parse(bad_json_str, stage="original")
    if data is not None:
        return data

    fixed_str = bad_json_str
    fixed_str = fixed_str.replace("\\0", "\\\\0").replace("'\0'", "'\\\\0'")
    fixed_str = re.sub(r'"\s*\n\s*"', '",\n"', fixed_str)
    fixed_str = re.sub(r'\\(?![ntr"\\u0])', r'\\\\', fixed_str)
    fixed_str = re.sub(r',\s*([}\]])', r'\1', fixed_str)
    if "'" in fixed_str and '"' not in fixed_str:
        fixed_str = fixed_str.replace("'", '"')

    data = attempt_parse(fixed_str, stage="after fixes")
    if data is not None:
        return data

    raise ValueError(f"Line {line_no}: JSON parsing failed after fixes.")

def load_database(jsonl_path):
    """加载 feature.jsonl，解析 analysis 字段"""
    database = []
    with jsonlines.open(jsonl_path, "r") as reader:
        for idx, obj in enumerate(tqdm(reader, desc="Loading database"), start=1):
            analysis_str = str(obj.get("analysis", "")).strip()
            if not analysis_str:
                continue

            # 🔎 用正则提取第一个 ```json ... ``` 代码块
            match = re.search(r"```json(.*?)```", analysis_str, re.DOTALL | re.IGNORECASE)
            if match:
                analysis_str = match.group(1).strip()

            # 尝试解析
            try:
                data = json.loads(analysis_str)
            except Exception:
                try:
                    data = fix_and_parse_json(analysis_str, line_no=idx)
                except Exception:
                    print(f"[SKIP] Skipping line {idx} because parsing failed.")
                    continue

            # 加入数据库
            if isinstance(data, list):
                for entry in data:
                    db_text = " ".join(entry.get("Unoptimized Code Conditions", []))
                    database.append({
                        "text": db_text,
                        "operation": entry.get("Optimization Operation", "")
                    })

    print(f"\n✅ Loaded {len(database)} entries from database")
    return database

def parse_optimized_features(raw_value, line_no=None):
    """解析 extract_feature.jsonl 中的 optimized_features 字段（支持字符串中间的 ```json 包裹）"""
    if isinstance(raw_value, str):
        s = raw_value.strip()
        if not s:
            return []

        # 🔎 用正则提取第一个 ```json ... ``` 代码块
        match = re.search(r"```json(.*?)```", s, re.DOTALL | re.IGNORECASE)
        if match:
            s = match.group(1).strip()

        # 尝试直接解析
        try:
            return json.loads(s)
        except Exception as e:
            print(f"[WARN] Failed to parse optimized_features at line {line_no}: {e}")
            # 尝试修复
            try:
                return fix_and_parse_json(s, line_no=line_no)
            except Exception:
                print(f"[SKIP] Skipping optimized_features at line {line_no}")
                return []

    elif isinstance(raw_value, list):
        return raw_value

    else:
        return []

def find_top_matches(feature, database, model, top_n=3):
    """找到单个新特征在数据库中最相似的前n个优化方案"""
    db_texts = [item["text"] for item in database]
    db_embeddings = model.encode(db_texts, convert_to_tensor=True)

    feature_text = " ".join(feature["Unoptimized Code Conditions"])
    feature_embedding = model.encode(feature_text, convert_to_tensor=True)

    cos_scores = util.cos_sim(feature_embedding, db_embeddings)[0]
    top_results = cos_scores.topk(top_n)

    matches = [database[idx]["operation"] for idx in top_results.indices]

    return {
        "Unoptimized Code Conditions": feature["Unoptimized Code Conditions"],
        "Optimization Operation": matches
    }

# ---------------- 主脚本 -------------------

if __name__ == "__main__":
    # 1. 加载数据库
    database = load_database("feature.jsonl")

    # 2. 加载 extract_feature.jsonl 的原始行
    all_objs = []
    with jsonlines.open("extract_feature.jsonl", "r") as reader:
        all_objs = list(reader)

    # 3. 加载模型
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    # 4. 逐行处理，生成 analysis
    with jsonlines.open("match.jsonl", "w") as writer:
        for idx, obj in enumerate(tqdm(all_objs, desc="Processing lines"), start=1):
            raw_optimized = obj.get("optimized_features", [])
            optimized_features = parse_optimized_features(raw_optimized, line_no=idx)

            matches_for_line = []
            if isinstance(optimized_features, list):
                for feat in optimized_features:
                    if isinstance(feat, dict) and "Unoptimized Code Conditions" in feat:
                        result = find_top_matches(feat, database, model, top_n=2)
                        matches_for_line.append(result)

            # analysis = JSON 数组（markdown 包裹）
            obj["analysis"] = "```json\n" + json.dumps(matches_for_line, indent=2, ensure_ascii=False) + "\n```"

            writer.write(obj)

    print("✅ Matching completed. Results saved to match.jsonl, each line now has analysis")