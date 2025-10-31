from underthesea import word_tokenize
from rank_bm25 import BM25Okapi
import re, unicodedata, json

def preprocess_query(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)  # bỏ dấu câu/ký tự đặc biệt
    s = re.sub(r"\s+", " ", s).strip()
    return s

def vi_tokenize(s: str):
    # format="text" -> chuỗi tokens (multiword có dấu '_'), rồi .split() thành list
    return word_tokenize(preprocess_query(s), format="text").split()

# (tuỳ chọn) stopwords
with open("vietnamese-stopwords.txt","r",encoding="utf-8") as f:
    STOPWORDS = set(f.read().splitlines())

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

query = "phải thực hiện thao tác nạp mẫu vào bình chứa và xử lý mẫu sơ bộ bằng hóa chất như thế nào?"

# 1) Đọc JSONL -> xây 2 list: ids và contents
ids = []
contents = []
with open("data/Retrieval/semantic_chunking_for_embedding.jsonl","r",encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        content = (r.get("content") or "").strip()
        if not content:
            continue
        article_id = r.get("article_id")
        clause_id  = r.get("clause_id")
        # Chuỗi ID theo yêu cầu: {article_id}#{clause_id}
        rid = f"{article_id}#{clause_id}"
        ids.append(rid)
        contents.append(content)

assert len(ids) == len(contents) and len(ids) > 0, "Corpus rỗng hoặc mapping lệch!"

# 2) Tokenize corpus (và lọc stopwords nếu muốn)
tokenized_corpus = [remove_stopwords(vi_tokenize(p)) for p in contents]

# 3) Khởi tạo BM25 trên toàn bộ corpus
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

# 4) Tokenize query (và lọc stopwords)
tokenized_query = remove_stopwords(vi_tokenize(query))

# 5) Lấy điểm & xếp hạng
scores = bm25.get_scores(tokenized_query)
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

# # 6) In top-1: ID theo format {article_id}#{clause_id} và nội dung
# top1_idx, top1_score = ranked[0]
# top1_id = ids[top1_idx]
# top1_content = contents[top1_idx]
# print(f"[TOP-1] id={top1_id} | score={top1_score:.4f}\n{top1_content}\n")

# (Tuỳ chọn) In top-5 kèm ID
print("=== Top-5 ===")
for rank, (i, sc) in enumerate(ranked[:5], 1):
    print(f"{rank}. id={ids[i]} | score={sc:.4f}")
