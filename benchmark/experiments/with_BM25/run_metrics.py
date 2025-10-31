import json
import sys
from pathlib import Path
from typing import Callable, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.evaluate_metric import (
    f1_score,
    hitrate_at_k,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

TOP_K = [1, 3, 5, 10]
RESULTS_INPUT_PATHS: Dict[int, Path] = {
    1: Path("data/Result_by_BM25/result_by_top1.jsonl"),
    3: Path("data/Result_by_BM25/result_by_top3.jsonl"),
    5: Path("data/Result_by_BM25/result_by_top5.jsonl"),
    10: Path("data/Result_by_BM25/result_by_top10.jsonl"),
}

GROUND_TRUTH_INPUT_PATH = Path("data/Retrieval/question_n_rel_id_enriched.jsonl")
RESULTS_OUTPUT_DIR = Path("benchmark/results_of_experiments/by_BM25")


MetricFn = Callable[[List[List[str]], List[List[str]], int], float]

METRIC_FUNCTIONS: Dict[str, MetricFn] = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "hitrate": hitrate_at_k,
    "map": map_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}

OUTPUT_FILENAMES: Dict[str, str] = {
    "precision": "precision.json",
    "recall": "recall.json",
    "hitrate": "hitrate.json",
    "map": "map.json",
    "mrr": "mrr.json",
    "ndcg": "NDCG.json",
    "f1score": "f1score.json",
}


def load_ground_truths(path: Path) -> List[List[str]]:
    ground_truths: List[List[str]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            try:
                answers = json.loads(line).get("answer") or []
            except json.JSONDecodeError:
                answers = []
            if isinstance(answers, (list, tuple, set)):
                ground_truths.append([str(item) for item in answers])
            else:
                ground_truths.append([])
    return ground_truths


def load_retrievals(path: Path) -> List[List[str]]:
    retrievals: List[List[str]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            try:
                answers = json.loads(line).get("answer") or []
            except json.JSONDecodeError:
                answers = []
            if isinstance(answers, (list, tuple, set)):
                retrievals.append([str(item) for item in answers])
            else:
                retrievals.append([])
    return retrievals


def write_metric_results(metric: str, scores: Dict[str, float]) -> None:
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_OUTPUT_DIR / OUTPUT_FILENAMES[metric]
    with output_file.open("w", encoding="utf-8") as outfile:
        json.dump(scores, outfile, ensure_ascii=False, indent=2)


def main() -> None:
    ground_truths = load_ground_truths(GROUND_TRUTH_INPUT_PATH)
    retrievals_by_top: Dict[int, List[List[str]]] = {}

    for top_k, path in RESULTS_INPUT_PATHS.items():
        retrievals = load_retrievals(path)
        if len(retrievals) != len(ground_truths):
            raise ValueError(
                f"Số lượng kết quả ({len(retrievals)}) không khớp với ground truth ({len(ground_truths)})"
            )
        retrievals_by_top[top_k] = retrievals

    computed_metrics: Dict[str, Dict[str, float]] = {}

    for metric_name, metric_fn in METRIC_FUNCTIONS.items():
        metric_scores: Dict[str, float] = {}
        for top_k in TOP_K:
            metric_scores[f"top_{top_k}"] = metric_fn(
                retrievals_by_top[top_k], ground_truths, top_k
            )
        computed_metrics[metric_name] = metric_scores
        write_metric_results(metric_name, metric_scores)

    if "precision" in computed_metrics and "recall" in computed_metrics:
        f1_scores: Dict[str, float] = {}
        for top_k in TOP_K:
            precision_value = computed_metrics["precision"][f"top_{top_k}"]
            recall_value = computed_metrics["recall"][f"top_{top_k}"]
            f1_scores[f"top_{top_k}"] = f1_score(precision_value, recall_value)

        computed_metrics["f1score"] = f1_scores
        write_metric_results("f1score", f1_scores)


if __name__ == "__main__":
    main()
