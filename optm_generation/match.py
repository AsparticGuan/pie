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

def extract_json_from_string(raw_str, line_no=None, field_name="field"):
    """
    从字符串中提取并解析 JSON（支持 markdown ```json 包裹）
    - 尝试提取代码块并解析
    - 失败时回退到原始字符串解析
    - 如果仍失败，返回原始字符串
    """
    if not isinstance(raw_str, str):
        return raw_str

    s = raw_str.strip()
    if not s:
        return raw_str

    # 🔎 尝试提取第一个 ```json ... ``` 代码块
    match = re.search(r"```json(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if match:
        inner = match.group(1).strip()
    else:
        inner = s

    # 尝试解析提取到的内容
    try:
        return json.loads(inner)
    except Exception as e:
        print(f"[WARN] Failed to parse code block from {field_name} at line {line_no}: {e}")
        # 尝试修复
        try:
            return fix_and_parse_json(inner, line_no=line_no)
        except Exception:
            print(f"[INFO] Trying raw string parse for {field_name} at line {line_no}")
            # 再试用原始字符串解析
            try:
                return json.loads(s)
            except Exception as e2:
                print(f"[FALLBACK] Returning raw {field_name} at line {line_no}: {e2}")
                return raw_str
        
def parse_optimized_features(raw_value, line_no=None):
    """解析 extract_feature.jsonl 中的 optimized_features 字段"""
    if isinstance(raw_value, list):
        return raw_value
    return extract_json_from_string(raw_value, line_no=line_no, field_name="optimized_features")

def load_database(jsonl_path):
    """加载 feature.jsonl，解析 analysis 字段"""
    database = []
    with jsonlines.open(jsonl_path, "r") as reader:
        for idx, obj in enumerate(tqdm(reader, desc="Loading database"), start=1):
            analysis = extract_json_from_string(obj.get("analysis", ""), line_no=idx, field_name="analysis")

            if isinstance(analysis, list):
                for entry in analysis:
                    db_text = " ".join(entry.get("Unoptimized Code Conditions", []))
                    database.append({
                        "text": db_text,
                        "operation": entry.get("Optimization Operation", "")
                    })
            else:
                # 字符串或其它 → 保留原始
                database.append({
                    "text": str(analysis),
                    "operation": ""
                })

    print(f"\n✅ Loaded {len(database)} entries from database")
    return database

def find_top_matches(feature, database, model, top_n=1):
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

            print(optimized_features)

            matches_for_line = []
            if isinstance(optimized_features, list):
                for feat in optimized_features:
                    if isinstance(feat, dict) and "Unoptimized Code Conditions" in feat:
                        result = find_top_matches(feat, database, model, top_n=1)
                        matches_for_line.append(result)

            # analysis = JSON 数组（markdown 包裹）
            obj["analysis"] = "```json\n" + json.dumps(matches_for_line, indent=2, ensure_ascii=False) + "\n```"

            writer.write(obj)

    print("✅ Matching completed. Results saved to match.jsonl, each line now has analysis")