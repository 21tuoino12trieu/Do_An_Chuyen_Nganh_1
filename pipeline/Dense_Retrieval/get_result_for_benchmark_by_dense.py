import json
import logging
import sys
from pathlib import Path
from typing import Iterator, List

from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from retrieval_by_dense_vector import DenseVectorRetriever

logger = logging.getLogger(__name__)

load_dotenv()

INPUT_QUESTION_PATH = Path("data/Retrieval/question_n_rel_id_enriched.jsonl")
OUTPUT_DIR = Path("data/Result_by_Dense")
TOP_K_VALUES: List[int] = [1, 3, 5, 10]
COLLECTION_NAMES = [
    "legal_clauses_AITeamVN",
    # "legal_clauses_jina-v3",
    # "legal_clauses_Qwen3",
    # "legal_clauses_vn_dcm_embedding",
]
MODEL_NAMES = [
    "AITeamVN",
    # "jina-embeddings-v3",
    # "Qwen3",
    # "vn_dcm_embedding",
]


def read_questions(path: Path) -> Iterator[str]:
    """Yield cleaned questions from the JSONL file."""

    with path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                question = json.loads(raw_line).get("question")
            except json.JSONDecodeError as exc:
                logger.warning("Bỏ qua dòng %d vì lỗi JSON: %s", line_number, exc)
                continue

            if isinstance(question, str) and question.strip():
                yield question.strip()
            else:
                logger.debug(
                    "Bỏ qua dòng %d vì không có câu hỏi hợp lệ trong %s",
                    line_number,
                    path,
                )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    for collection_name, model_name in zip(COLLECTION_NAMES, MODEL_NAMES):
        retriever = DenseVectorRetriever(collection_name=collection_name)
        for top_k in TOP_K_VALUES:
            output_path = OUTPUT_DIR / model_name / f"result_by_top{top_k}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            count = 0

            with output_path.open("w", encoding="utf-8") as outfile:
                for question in read_questions(INPUT_QUESTION_PATH):
                    results = retriever.search(query=question, top_k=top_k)
                    answer_ids = [
                        f"{res.get('article_id')}#{res.get('clause_id')}"
                        for res in results
                        if res.get("article_id") and res.get("clause_id")
                    ]
                    payload = {"question": question, "answer": answer_ids}
                    outfile.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    count += 1

            logger.info("Đã ghi %d kết quả vào %s", count, output_path)


if __name__ == "__main__":
    main()
