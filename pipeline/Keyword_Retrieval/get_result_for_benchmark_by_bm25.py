import json
import logging
import sys
from pathlib import Path
from typing import Iterator, List

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from retrieval_by_bm25 import BM25Retriever

logger = logging.getLogger(__name__)

DOCUMENTS_PATH = Path("data/Retrieval/semantic_chunking_for_embedding.jsonl")
INPUT_QUESTION_PATH = Path("data/Retrieval/question_n_rel_id_enriched.jsonl")
OUTPUT_DIR = Path("data/Result_by_BM25")
TOP_K_VALUES: List[int] = [1, 3, 5, 10]


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
                    "Bỏ qua dòng %d vì không có câu hỏi hợp lệ trong %s", line_number, path
                )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    retriever = BM25Retriever(documents_path=str(DOCUMENTS_PATH))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for top_k in TOP_K_VALUES:
        output_path = OUTPUT_DIR / f"result_by_top{top_k}.jsonl"
        count = 0

        with output_path.open("w", encoding="utf-8") as outfile:
            for question in read_questions(INPUT_QUESTION_PATH):
                result = retriever.search(question, top_k=top_k)
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1

        logger.info("Đã ghi %d kết quả vào %s", count, output_path)


if __name__ == "__main__":
    main()
