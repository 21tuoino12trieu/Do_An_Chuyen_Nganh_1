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
MODEL_NAMES = ["AITeamVN", "jina-embeddings-v3", "Qwen3", "vn_dcm_embedding"]
RESULTS_INPUT_PATHS: Dict[str, Dict[int, Path]] = {
    model_name: {
        top_k: Path(f"data/Result_by_Dense/{model_name}/result_by_top{top_k}.jsonl")
        for top_k in TOP_K
    }
    for model_name in MODEL_NAMES
}


GROUND_TRUTH_INPUT_PATH = Path("data/Retrieval/question_n_rel_id_enriched.jsonl")
RESULTS_OUTPUT_DIR = Path("benchmark/results_of_experiments/by_Dense")

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

def write_metric_results(metric: str, scores: Dict[str, Dict[str, float]]) -> None:
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_OUTPUT_DIR / OUTPUT_FILENAMES[metric]
    with output_file.open("w", encoding="utf-8") as outfile:
        json.dump(scores, outfile, ensure_ascii=False, indent=2)

def main() -> None:
    ground_truths = load_ground_truths(GROUND_TRUTH_INPUT_PATH)
    retrievals_by_model: Dict[str, Dict[int, List[List[str]]]] = {}

    for model_name, paths_by_top in RESULTS_INPUT_PATHS.items():
        retrievals_by_top: Dict[int, List[List[str]]] = {}
        for top_k, path in paths_by_top.items():
            retrievals = load_retrievals(path)
            if len(retrievals) != len(ground_truths):
                raise ValueError(
                    f"Số lượng kết quả ({len(retrievals)}) không khớp với ground truth ({len(ground_truths)})"
                )
            retrievals_by_top[top_k] = retrievals
        retrievals_by_model[model_name] = retrievals_by_top

    computed_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for metric_name, metric_fn in METRIC_FUNCTIONS.items():
        metric_scores_by_model: Dict[str, Dict[str, float]] = {}
        for model_name in MODEL_NAMES:
            metric_scores: Dict[str, float] = {}
            for top_k in TOP_K:
                metric_scores[f"top_{top_k}"] = metric_fn(
                    retrievals_by_model[model_name][top_k], ground_truths, top_k
                )
            metric_scores_by_model[model_name] = metric_scores
        computed_metrics[metric_name] = metric_scores_by_model
        write_metric_results(metric_name, metric_scores_by_model)

    if "precision" in computed_metrics and "recall" in computed_metrics:
        f1_scores_by_model: Dict[str, Dict[str, float]] = {}
        for model_name in MODEL_NAMES:
            precision_scores = computed_metrics["precision"][model_name]
            recall_scores = computed_metrics["recall"][model_name]
            model_f1_scores: Dict[str, float] = {}
            for top_k in TOP_K:
                key = f"top_{top_k}"
                model_f1_scores[key] = f1_score(
                    precision_scores[key], recall_scores[key]
                )
            f1_scores_by_model[model_name] = model_f1_scores

        computed_metrics["f1score"] = f1_scores_by_model
        write_metric_results("f1score", f1_scores_by_model)


if __name__ == "__main__":
    main()
