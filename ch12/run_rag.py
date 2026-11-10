# 12 章の RAG のコードを本文から取り出し、架空の社内文書 2 本でつないで動かす
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from blocks import block
import re, pathlib, json, numpy as np, shutil
get = lambda key: block("12", key)

# ①〜③ 取り込み・分割・埋め込み・格納
exec(get("def split_markdown"))
chunks = []
for p in sorted((pathlib.Path(__file__).resolve().parent / "data" / "rag_corpus").glob("*.md")):
    chunks += split_markdown(p.read_text(encoding="utf-8"), p.name)
print("チャンク数", len(chunks)); print(" 例:", chunks[0]["heading"], "/", chunks[0]["raw"][:30])
exec(get("def embed_passages"))
from transformers import AutoTokenizer
etok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
sample = "".join(c["raw"] for c in chunks)
print(f"日本語 1 文字あたりのトークン数（e5 のトークナイザ）: {len(etok(sample, add_special_tokens=False)['input_ids']) / len(sample):.2f}")
shutil.rmtree("index", ignore_errors=True)
blk = get("import lancedb, pyarrow as pa")
head, idx = blk.split("from lancedb.index", 1)
exec(head)                                             # 表を作って追加するまで
try:
    exec(blk[blk.index("from lancedb.index"):blk.index("# 全文検索")])   # 小さい表での IVF_PQ
    print("create_index: 成功")
except Exception as e:
    print("create_index（行数", tbl.count_rows(), "）:", type(e).__name__, str(e)[:120])
exec("tbl.create_fts_index" + idx.split("tbl.create_fts_index", 1)[1])
print("FTS 索引: 作成")

# ④⑤ ハイブリッド検索・RRF・リランキング（vector_search・bm25_search も本文の定義）
for q in ("グリーン車は会社が払ってくれますか", "USB メモリ"):
    exec(get("def rrf"))
    print(f"質問「{q}」 ベクトル上位:", vec_ids[:2], " BM25 上位:", bm25_ids[:2], " RRF 上位:", fused[:2])
exec(get("from sentence_transformers import CrossEncoder"))
q = "グリーン車は会社が払ってくれますか"
by_id = {c["id"]: c for c in chunks}
exec(get("def rrf"))
top = rerank(q, [by_id[i] for i in fused], top_n=3)
for d in top: print(f"  rerank {d['score']:.3f} {d['heading']}")
print("関係のない質問での rerank 件数:", len(rerank("今日の天気は", [by_id[i] for i in fused], top_n=3)))

# ⑥ プロンプト・引用の検査
exec(get("SYSTEM = "))
msgs = build_prompt(q, top)
print("プロンプトの doc 数:", msgs[1]["content"].count("<doc id="))
exec(get("def verify_citations"))
good = f"グリーン車の差額は自己負担です[{top[0]['id']}]。"
bad  = "全額会社負担です[0123456789abcdef]。指定席までです。"
print("引用の検査（正しい答え）:", verify_citations(good, top)["hallucinated_ids"])
v = verify_citations(bad, top)
print("引用の検査（捏造あり）:", v["hallucinated_ids"], " 引用なしの文:", v["uncited_sentences"])

# 評価
exec(get("def retrieval_metrics"))
gold = {next(c["id"] for c in chunks if "新幹線" in c["heading"])}
def retrieve(q, top_k=10):
    return [by_id[i] for i in rrf([[h["id"] for h in vector_search(q, 50)], [h["id"] for h in bm25_search(q, 50)]])[:top_k]]
print("検索の指標:", {k: round(float(v), 3) for k, v in retrieval_metrics([{"q": q, "gold": gold}], retrieve, k=5).items()})
exec(get("def faithfulness"))
judge = lambda prompt: "supported" if "自己負担" in prompt.split("主張:")[1] else "unsupported"
print("忠実性:", faithfulness("グリーン車の差額は自己負担です。宿泊は無制限です。", top, judge))
exec(get("def classify_failure"))
print("失敗の分類:", classify_failure({"gold": gold}, top, top, True, 1.0))

# 前処理フィルタ（tenant_id・updated_at の列を持つ表で）
tbl2 = db.create_table("docs_mt", [{**c, "vector": v, "tenant_id": ("a" if i % 2 else "b"), "updated_at": "2026-0" + str(i + 1) + "-01"}
                                   for i, (c, v) in enumerate(zip(chunks, embed_passages([c["text"] for c in chunks])))], mode="overwrite")
tbl_backup, tbl = tbl, tbl2
tenant, since = "a", "2026-02-01"
exec(get(".where(f\"tenant_id"))
print("前処理フィルタ:", [(h["tenant_id"], h["updated_at"]) for h in hits])
