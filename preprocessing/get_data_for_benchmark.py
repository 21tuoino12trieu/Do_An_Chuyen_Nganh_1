import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# Bảo đảm có thể import prompt.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompt.prompt_for_rematting_qa import PROMPT  # noqa: E402


def load_triplets(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("API_KEY")

    client = OpenAI(base_url=base_url, api_key=api_key)

    input_path = PROJECT_ROOT / "data" / "question_n_ans_rel.jsonl"
    output_path = PROJECT_ROOT / "data" / "question_n_rel_id.jsonl"

    triplets = load_triplets(input_path)
    total = len(triplets)

    grouped_answers: Dict[str, set] = {}

    for idx, triplet in enumerate(triplets, start=1):
        user_question = triplet.get("query", "")
        positive_id = triplet.get("positive_id", "")
        response_law = triplet.get("content", "")

        combined_input = (
            f"user_question: {user_question}\n"
            f"positive_id: {positive_id}\n"
            f"response_law: {response_law}"
        )

        response = client.chat.completions.create(
            model="misa-qwen3-235b",
            messages=[
                {"role": "system", "content": PROMPT["FINDING_RELEVANT_ANSWERS"]},
                {"role": "user", "content": combined_input},
            ],
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content.strip()
        try:
            parsed_answer = json.loads(raw_content)
        except json.JSONDecodeError:
            parsed_answer = {"raw": raw_content}

        key = parsed_answer.get("question", user_question)
        answers = parsed_answer.get("answer", [])
        print(answers)
        if isinstance(answers, list):
            grouped_answers.setdefault(key, set()).update(answers)

        print(f"Processed {idx}/{total}", flush=True)

    with output_path.open("w", encoding="utf-8") as outfile:
        for question, answers in grouped_answers.items():
            record = {
                "question": question,
                "answer": sorted(answers),
            }
            outfile.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Finished {total} questions (merged duplicates).", flush=True)


if __name__ == "__main__":
    main()
