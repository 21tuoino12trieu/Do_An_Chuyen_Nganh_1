import difflib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_PATH = DATA_DIR / "question_n_ans_rel.jsonl"
TARGET_PATH = DATA_DIR / "question_n_rel_id.jsonl"
OUTPUT_PATH = DATA_DIR / "question_n_rel_id_enriched.jsonl"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompt.prompt_for_rematting_qa import PROMPT  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def call_chat_completion(
    client: OpenAI,
    model_sequence: Sequence[str],
    prompt_text: str,
    combined_input: str,
    retryable_statuses: Set[int],
    max_attempts: int,
    base_delay: float,
    log_label: str,
) -> str:
    for model_idx, model_name in enumerate(model_sequence):
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": prompt_text},
                        {"role": "user", "content": combined_input},
                    ],
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content.strip()
            except OpenAIError as exc:
                status_code = getattr(exc, "status_code", None)
                non_retryable = status_code is not None and status_code not in retryable_statuses
                if non_retryable:
                    raise

                delay = base_delay * (2 ** (attempt - 1))
                print(
                    f"[WARN] {log_label}: retrying with model '{model_name}' in {delay:.1f}s "
                    f"after {exc.__class__.__name__} (status={status_code})",
                    file=sys.stderr,
                    flush=True,
                )
                if delay:
                    time.sleep(delay)

                if attempt == max_attempts:
                    if model_idx == len(model_sequence) - 1:
                        raise
                    next_model = model_sequence[model_idx + 1]
                    print(
                        f"[INFO] {log_label}: switching to fallback model '{next_model}'.",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception:
                raise

    raise RuntimeError(f"{log_label}: all models in sequence exhausted without success.")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("API_KEY")

    if not api_key:
        raise RuntimeError("Missing API key. Set API_KEY in .env or export OPENAI_API_KEY.")

    client = OpenAI(base_url=base_url, api_key=api_key)

    max_attempts = max(1, int(os.getenv("OPENAI_MAX_RETRIES", "5")))
    base_delay = max(0.0, float(os.getenv("OPENAI_RETRY_DELAY", "1.0")))
    retryable_statuses = {408, 409, 429, 500, 502, 503, 504}

    primary_model = os.getenv("OPENAI_PRIMARY_MODEL", "misa-qwen3-235b")
    fallback_models = [
        model.strip()
        for model in os.getenv("OPENAI_FALLBACK_MODELS", "misa-gpt-oss-120b").split(",")
        if model.strip()
    ]
    model_sequence: List[str] = []
    seen_models: Set[str] = set()
    for candidate in [primary_model, *fallback_models]:
        if candidate and candidate not in seen_models:
            model_sequence.append(candidate)
            seen_models.add(candidate)
    if not model_sequence:
        raise ValueError("No model specified. Set OPENAI_PRIMARY_MODEL or OPENAI_FALLBACK_MODELS.")

    source_records = load_jsonl(SOURCE_PATH)
    target_records = load_jsonl(TARGET_PATH)

    question_lookup: Dict[str, Dict[str, Any]] = {}
    for source in source_records:
        question = source.get("query")
        if question and question not in question_lookup:
            question_lookup[question] = source
    source_questions = list(question_lookup.keys())

    prompt_text = PROMPT["FINDING_RELEVANT_ANSWERS"]

    total_targets = len(target_records)
    iteration = 0

    while True:
        pending = [
            (idx, record)
            for idx, record in enumerate(target_records, start=1)
            if not record.get("answer")
        ]
        if not pending:
            break

        iteration += 1
        total_pending = len(pending)
        print(
            f"Iteration {iteration}: {total_pending} / {total_targets} records still empty.",
            flush=True,
        )

        iteration_progress = False

        for position, (idx, target) in enumerate(pending, start=1):
            question = target.get("question", "")
            source = question_lookup.get(question)
            matched_question = question
            if not source:
                close_matches = difflib.get_close_matches(question, source_questions, n=1, cutoff=0.9)
                if close_matches:
                    matched_question = close_matches[0]
                    source = question_lookup.get(matched_question)
                    print(
                        f"[INFO] {log_label}: using closest match '{matched_question}' for question lookup.",
                        file=sys.stderr,
                        flush=True,
                    )
            if not source:
                raise RuntimeError(
                    f"Question at index {idx} not found in source file; cannot populate answer."
                )

            user_question = source.get("query", matched_question)
            positive_id = source.get("positive_id", "")
            response_law = source.get("content", "")

            combined_input = (
                f"user_question: {user_question}\n"
                f"positive_id: {positive_id}\n"
                f"response_law: {response_law}"
            )

            log_label = f"iter {iteration} {idx}/{total_targets} ({position}/{total_pending})"
            raw_content = call_chat_completion(
                client=client,
                model_sequence=model_sequence,
                prompt_text=prompt_text,
                combined_input=combined_input,
                retryable_statuses=retryable_statuses,
                max_attempts=max_attempts,
                base_delay=base_delay,
                log_label=log_label,
            )

            try:
                parsed = json.loads(raw_content)
            except json.JSONDecodeError:
                print(
                    f"[WARN] {log_label}: response is not valid JSON, storing raw content.",
                    file=sys.stderr,
                    flush=True,
                )
                parsed = {"answer": [], "raw": raw_content}

            answers = parsed.get("answer", [])
            if isinstance(answers, list):
                cleaned_answers = sorted({str(item) for item in answers if item})
                target["answer"] = cleaned_answers
                if cleaned_answers:
                    iteration_progress = True
            else:
                print(
                    f"[WARN] {log_label}: 'answer' field missing or not a list; leaving empty.",
                    file=sys.stderr,
                    flush=True,
                )
                target["answer"] = []

            print(
                f"Filled question {idx}/{total_targets} "
                f"({position}/{total_pending} needing answers).",
                flush=True,
            )

        write_jsonl(OUTPUT_PATH, target_records)

        if not iteration_progress:
            raise RuntimeError(
                "No answers were populated in this iteration. Stopping to avoid endless loop."
            )

    print(f"Done. Updated file written to {OUTPUT_PATH}.", flush=True)


if __name__ == "__main__":
    main()
