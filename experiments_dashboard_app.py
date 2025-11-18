import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

RESULTS_ROOT = Path(__file__).resolve().parent / "benchmark" / "results_of_experiments"
BY_BM25 = RESULTS_ROOT / "by_BM25"
BY_DENSE = RESULTS_ROOT / "by_Dense"
BY_DENSE_RERANK = RESULTS_ROOT / "by_Dense_Rerank"
TEXT_HIGHLIGHT_COLOR = "#d32f2f"
TABLE_CMAP = "PuBu"

st.set_page_config(
    page_title="So sánh kết quả thí nghiệm",
    page_icon="📊",
    layout="wide",
)

st.title("📈 Bảng so sánh mô hình & phương pháp đánh giá")

def _discover_metrics() -> List[str]:
    metric_names = set()
    for directory in (BY_BM25, BY_DENSE, BY_DENSE_RERANK):
        if not directory.exists():
            continue
        metric_names.update(path.stem for path in directory.glob("*.json"))
    return sorted(metric_names)


def _format_topk(key: str) -> str:
    if key.startswith("top_"):
        return f"Top@{key.split('_', 1)[1]}"
    return key


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def _highlight_max_text(data: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    for column in data.columns:
        max_value = data[column].max()
        mask = data[column] == max_value
        styles.loc[mask, column] = f"font-weight: 700; color: {TEXT_HIGHLIGHT_COLOR};"
    return styles


@st.cache_data(show_spinner=False)
def load_metric_table(metric_name: str) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, float]] = []
    warnings: List[str] = []

    dense_file = BY_DENSE / f"{metric_name}.json"
    if dense_file.exists():
        dense_payload = _load_json(dense_file)
        for model_name, scores in dense_payload.items():
            row: Dict[str, float] = {"Method": "Dense", "Model": model_name}
            for key, value in scores.items():
                row[_format_topk(key)] = float(value)
            rows.append(row)
    else:
        warnings.append(f"Không tìm thấy file {dense_file}")

    dense_rerank_file = BY_DENSE_RERANK / f"{metric_name}.json"
    if dense_rerank_file.exists():
        dense_rerank_payload = _load_json(dense_rerank_file)
        for model_name, scores in dense_rerank_payload.items():
            row = {"Method": "Dense_Rerank", "Model": model_name}
            for key, value in scores.items():
                row[_format_topk(key)] = float(value)
            rows.append(row)
    else:
        warnings.append(f"Không tìm thấy file {dense_rerank_file}")

    bm25_file = BY_BM25 / f"{metric_name}.json"
    if bm25_file.exists():
        bm25_payload = _load_json(bm25_file)
        row = {"Method": "BM25", "Model": "baseline"}
        for key, value in bm25_payload.items():
            row[_format_topk(key)] = float(value)
        rows.append(row)
    else:
        warnings.append(f"Không tìm thấy file {bm25_file}")

    if not rows:
        return pd.DataFrame(), warnings

    df = pd.DataFrame(rows)
    df = df.set_index(["Method", "Model"]).sort_index()
    topk_columns = sorted(
        (col for col in df.columns if col.startswith("Top@")),
        key=lambda name: int(name.split("@", 1)[1]),
    )
    df = df[topk_columns]
    return df, warnings


metric_options = _discover_metrics()
if not metric_options:
    st.error(
        "Không tìm thấy bất kỳ file số liệu nào trong "
        "`benchmark/results_of_experiments/by_BM25` hoặc `by_Dense`hoặc `by_Dense_Rerank`. "
        "Hãy chạy benchmark trước khi mở giao diện này."
    )
    st.stop()

selected_metric = st.selectbox("Chọn độ đo để hiển thị", metric_options, index=0)

table, warning_messages = load_metric_table(selected_metric)
for warn in warning_messages:
    st.warning(warn)

if table.empty:
    st.error("Không có dữ liệu để hiển thị cho độ đo đã chọn.")
    st.stop()

topk_columns = list(table.columns)
styled_table = (
    table.style.format("{:.4f}")
    .apply(_highlight_max_text, subset=topk_columns, axis=None)
)

st.subheader(f"Bảng giá trị top@K cho độ đo **{selected_metric}**")
st.dataframe(styled_table, use_container_width=True)
