import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, Iterable, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
PROMPT_TEMPLATE = """Bạn là trợ lý pháp lý.
Hãy đọc câu hỏi và danh sách các khoản (clause) của điều luật {article_id}.
Chọn ra những clause trả lời trực tiếp câu hỏi; nếu không clause nào phù hợp hãy trả về danh sách rỗng.
Chỉ trả lời bằng JSON đúng định dạng:
{{"answer_clause_ids": ["1","2"]}}
Câu hỏi: {question}
Danh sách clause:
{clauses}
"""
def load_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
def group_clauses_by_article(records: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for record in records:
        article_id = record.get("article_id")
        if not article_id:
            continue
        grouped.setdefault(article_id, []).append(record)
    for clauses in grouped.values():
        clauses.sort(
            key=lambda item: (
                int(item.get("clause_id"))
                if isinstance(item.get("clause_id"), str)
                and item.get("clause_id").isdigit()
                else item.get("clause_id")
            )
        )
    return grouped
def build_prompt(question: str, article_id: str, clauses: List[dict]) -> str:
    clause_lines = []
    for clause in clauses:
        clause_id = clause.get("clause_id", "NA")
        content = clause.get("content", "").replace("\n", " ").strip()
        clause_lines.append(f"[{clause_id}] {content}")
    clauses_formatted = "\n".join(clause_lines)
    return PROMPT_TEMPLATE.format(
        question=question.strip(),
        article_id=article_id,
        clauses=clauses_formatted,
    )
def init_model(model_path: str, device: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if device is None else None,
        torch_dtype=torch_dtype,
    )
    if device is not None:
        model.to(device)
    return tokenizer, model
def generate_response(
    tokenizer,
    model,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    input_length = inputs["input_ids"].shape[-1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    generated = outputs[0][input_length:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()
def parse_clause_ids(raw_response: str, available_ids: List[str]) -> List[str]:
    candidate_ids: List[str] = []
    json_match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
    if json_match:
        json_text = json_match.group(0)
        try:
            data = json.loads(json_text)
            ids = data.get("answer_clause_ids") or data.get("clauses")
            if isinstance(ids, list):
                candidate_ids.extend(str(item).strip() for item in ids)
        except json.JSONDecodeError:
            pass
    if not candidate_ids:
        candidate_ids = re.findall(r"\b\d+\b", raw_response)
    filtered = []
    for clause_id in candidate_ids:
        clause_id = clause_id.strip()
        if clause_id in available_ids and clause_id not in filtered:
            filtered.append(clause_id)
    return filtered
def main():
    parser = argparse.ArgumentParser(
        description="Sử dụng LLM để chọn clause phù hợp với từng câu hỏi."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Đường dẫn tới mô hình LLM (HuggingFace hoặc local).",
    )
    parser.add_argument(
        "--triplets-path",
        default=os.path.join("Do_An_Chuyen_Nganh_1", "data", "triplets.jsonl"),
        help="Đường dẫn tới file JSONL chứa câu hỏi (query).",
    )
    parser.add_argument(
        "--semantic-path",
        default=os.path.join("Do_An_Chuyen_Nganh_1", "data", "semantic_chunking.jsonl"),
        help="Đường dẫn tới file JSONL chứa các clause.",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join("Do_An_Chuyen_Nganh_1", "data", "llm_selected_clauses.jsonl"),
        help="File JSONL đầu ra.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=4096,
        help="Giới hạn token cho phần prompt (cắt bớt nếu vượt quá).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Số token sinh thêm tối đa.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Nhiệt độ khi sinh văn bản (0 nghĩa là greedy).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (dùng khi temperature > 0).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Ví dụ "cuda:0" hoặc "cpu". Nếu bỏ trống sẽ để device_map="auto".',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số câu hỏi để chạy (debug).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logging.info("Đang tải dữ liệu clause ...")
    clauses = load_jsonl(args.semantic_path)
    article_to_clauses = group_clauses_by_article(clauses)
    logging.info("Đang tải câu hỏi ...")
    triplets = load_jsonl(args.triplets_path)
    if args.limit is not None:
        triplets = triplets[: args.limit]
    logging.info("Đang nạp mô hình %s ...", args.model_path)
    tokenizer, model = init_model(args.model_path, device=args.device)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    processed = 0
    with open(args.output_path, "w", encoding="utf-8") as outfile:
        for entry in triplets:
            question = entry.get("query", "").strip()
            article_id = entry.get("positive_id")
            if not question or not article_id:
                continue
            clause_candidates = article_to_clauses.get(article_id)
            if not clause_candidates:
                logging.warning("Không tìm thấy clause cho article_id %s", article_id)
                continue
            prompt = build_prompt(question, article_id, clause_candidates)
            response = generate_response(
                tokenizer,
                model,
                prompt,
                args.max_input_tokens,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
            )
            clause_ids = parse_clause_ids(
                response, [clause["clause_id"] for clause in clause_candidates]
            )
            selected_clauses = [
                {
                    "clause_id": clause["clause_id"],
                    "content": clause["content"],
                }
                for clause in clause_candidates
                if clause["clause_id"] in clause_ids
            ]
            record = {
                "question": question,
                "positive_id": article_id,
                "candidate_clause_ids": clause_ids,
                "clauses": selected_clauses,
                "llm_response": response,
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1
            if processed % 50 == 0:
                logging.info("Đã xử lý %d câu hỏi", processed)
    logging.info("Hoàn thành! Đã ghi %d dòng vào %s", processed, args.output_path)
if __name__ == "__main__":
    main()