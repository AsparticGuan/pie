import json
import jsonlines
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch

# ---------------- 工具函数 -------------------

def fix_and_parse_json(bad_json_str, line_no=None):
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
    if not isinstance(raw_str, str):
        return raw_str
    s = raw_str.strip()
    if not s:
        return raw_str

    match = re.search(r"```json(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if match:
        inner = match.group(1).strip()
    else:
        inner = s

    try:
        return json.loads(inner)
    except Exception:
        try:
            return fix_and_parse_json(inner, line_no=line_no)
        except Exception:
            try:
                return json.loads(s)
            except Exception:
                return raw_str
        
def parse_optimized_features(raw_value, line_no=None):
    if isinstance(raw_value, list):
        return raw_value
    return extract_json_from_string(raw_value, line_no=line_no, field_name="optimized_features")

def load_database(jsonl_path):
    database = []
    with jsonlines.open(jsonl_path, "r") as reader:
        for idx, obj in enumerate(tqdm(reader, desc="Loading database"), start=1):
            analysis = extract_json_from_string(obj.get("analysis", ""), line_no=idx, field_name="analysis")

            if isinstance(analysis, list):
                for entry in analysis:
                    # 支持两种字段
                    unopt = entry.get("Unoptimized Code Conditions") or entry.get("Unoptimized Code Condition") or []
                    if isinstance(unopt, str):
                        unopt = [unopt]
                    db_text = " ".join(unopt)
                    database.append({
                        "text": db_text,
                        "operation": entry.get("Optimization Operation", "")
                    })
            else:
                database.append({
                    "text": str(analysis),
                    "operation": ""
                })

    print(f"\n✅ Loaded {len(database)} entries from database")
    return database

# ---------------- 主脚本 -------------------

if __name__ == "__main__":
    # 1. 加载数据库
    database = load_database("feature.jsonl")
    db_texts = [item["text"] for item in database]

    # 2. 加载 extract_feature.jsonl
    with jsonlines.open("extract_feature.jsonl", "r") as reader:
        all_objs = list(reader)

    # 3. 收集所有待匹配 features
    all_features = []
    feature_to_obj_map = []  # (obj_index, local_feat_index)
    for obj_idx, obj in enumerate(all_objs, start=0):
        raw_optimized = obj.get("optimized_features", [])
        optimized_features = parse_optimized_features(raw_optimized, line_no=obj_idx+1)
        if isinstance(optimized_features, list):
            for f_idx, feat in enumerate(optimized_features):
                if isinstance(feat, dict):
                    unopt = feat.get("Unoptimized Code Conditions") or feat.get("Unoptimized Code Condition")
                    if unopt:
                        if isinstance(unopt, str):
                            unopt = [unopt]
                        all_features.append(" ".join(unopt))
                        feature_to_obj_map.append((obj_idx, f_idx))

    print(f"✅ Total features to match: {len(all_features)}")

    # 4. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=device)

    # 5. 计算数据库 embedding
    db_embeddings = model.encode(db_texts, convert_to_tensor=True, show_progress_bar=True)

    # 6. 批量计算 feature embedding
    feature_embeddings = model.encode(all_features, convert_to_tensor=True, show_progress_bar=True, batch_size=128)

    # 7. 相似度搜索（全部对比 → top1 with threshold）
    cos_scores = util.cos_sim(feature_embeddings, db_embeddings)  # shape: [num_features, num_db]
    top_results = torch.topk(cos_scores, k=1, dim=1)

    SIM_THRESHOLD = 0.8  # 👈 阈值，可以按需求调整

    # 8. 回填结果到原始对象
    per_obj_matches = [[] for _ in all_objs]
    for feat_idx, (obj_idx, local_idx) in enumerate(feature_to_obj_map):
        score = top_results.values[feat_idx][0].item()
        if score >= SIM_THRESHOLD:
            db_idx = top_results.indices[feat_idx][0].item()
            per_obj_matches[obj_idx].append({
                "Unoptimized Code Conditions": all_features[feat_idx],
                "Optimization Operation": [database[db_idx]["operation"]],
                # "SimilarityScore": round(score, 4)  # 可选：记录分数
            })
        else:
            # 没超过阈值 → 空匹配
            per_obj_matches[obj_idx].append({
                "Unoptimized Code Conditions": all_features[feat_idx],
                "Optimization Operation": [],
                # "SimilarityScore": round(score, 4)
            })

    # 把 matches 写回分析字段
    for obj_idx, obj in enumerate(all_objs):
        obj["analysis"] = "```json\n" + json.dumps(per_obj_matches[obj_idx], ensure_ascii=False) + "\n```"

    # 9. 保存结果
    with jsonlines.open("match.jsonl", "w") as writer:
        for obj in all_objs:
            writer.write(obj)

    print("✅ GPU batch matching completed with threshold. Results saved to match.jsonl")