import json
import logging
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

try:
    from rank_bm25 import BM25Okapi
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "rank_bm25 is required for BM25 retrieval. Install it with 'pip install rank-bm25'."
    ) from exc


logger = logging.getLogger(__name__)


def default_tokenizer(text: str) -> List[str]:
    """Lower-case and split text into whitespace tokens, keeping letters/digits."""

    if not text:
        return []

    normalized = text.lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return [token for token in normalized.split() if token]


class BM25Retriever:
    """Wrapper around rank_bm25.BM25Okapi for JSONL corpora."""

    def __init__(
        self,
        data_path: str,
        content_field: str = "content",
        tokenizer: Callable[[str], Sequence[str]] = default_tokenizer,
    ) -> None:
        self.data_path = Path(data_path)
        self.content_field = content_field
        self.tokenizer = tokenizer

        if not self.data_path.exists():
            logger.error("BM25 data file not found: %s", self.data_path)
            raise FileNotFoundError(f"BM25 data file not found: {self.data_path}")

        self.records: List[Dict] = []
        self.doc_tokens: List[List[str]] = []

        self._load_corpus()
        self._bm25 = BM25Okapi(self.doc_tokens)

        logger.info("BM25Okapi index loaded with %d documents", len(self.records))

    @property
    def num_documents(self) -> int:
        return len(self.records)

    def _load_corpus(self) -> None:
        with self.data_path.open("r", encoding="utf-8") as jsonl_file:
            for line_number, raw_line in enumerate(jsonl_file, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    logger.debug("Skipping empty line %d in %s", line_number, self.data_path)
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    logger.warning("Failed to parse line %d: %s", line_number, exc)
                    continue

                text = record.get(self.content_field, "")
                tokens = list(self.tokenizer(text))
                if not tokens:
                    logger.debug(
                        "Skipping document on line %d due to empty token list", line_number
                    )
                    continue

                self.records.append(record)
                self.doc_tokens.append(tokens)

        if not self.records:
            logger.error("No valid documents loaded from %s", self.data_path)
            raise ValueError(f"No valid documents loaded from {self.data_path}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
        if top_k <= 0:
            logger.error("top_k must be positive, received %s", top_k)
            raise ValueError("top_k must be positive")

        query_tokens = list(self.tokenizer(query))
        if not query_tokens:
            logger.warning("Query produced no tokens, returning empty result")
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

        results: List[Tuple[float, Dict]] = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])  # numpy float -> python float
            if score <= 0:
                continue
            results.append((score, self.records[idx]))

        logger.info(
            "BM25 retrieved %d documents (requested top_k=%d)",
            len(results),
            top_k,
        )
        return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="BM25 search over JSONL documents")
    parser.add_argument("--data", required=True, help="Path to JSONL file")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--field", default="content", help="Field containing document text")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    retriever = BM25Retriever(
        data_path=args.data,
        content_field=args.field,
    )

    results = retriever.search(args.query, top_k=args.top_k)
    for rank, (score, record) in enumerate(results, start=1):
        print(f"[{rank}] score={score:.4f} -> {record}")


if __name__ == "__main__":
    main()
