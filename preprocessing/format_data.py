import json
from pathlib import Path

TRIPLETS_PATH = Path("data/triplets.jsonl")
DOCUMENTS_PATH = Path("data/documents.jsonl")
OUTPUT_PATH = Path("data/question_n_ans_rel.jsonl")


def load_jsonl(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def main():
    triplets = load_jsonl(TRIPLETS_PATH)
    documents = load_jsonl(DOCUMENTS_PATH)

    doc_map = {}
    for doc in documents:
        doc_id = doc.get("id")
        if doc_id:
            doc_map[doc_id] = doc.get("document")

    for triplet in triplets:
        positive_id = triplet.get("positive_id")
        if not positive_id:
            continue
        content = doc_map.get(positive_id)
        if content:
            triplet["content"] = content

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as outfile:
        for triplet in triplets:
            outfile.write(json.dumps(triplet, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
