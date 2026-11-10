# 8 章の本文の Outlines ブロック（2 本）を、そのまま取り出して実行する
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from blocks import block
get = lambda key: block("08", key)
import re, pathlib
prompt = "次の問い合わせを分類して JSON で答えてください：ログイン画面で 500 エラーが出ます。"
exec(get("outlines.from_transformers"))
print("Ticket:", ticket.category, ticket.severity.value, ticket.needs_human)
exec(get("def get_generator"))
for _ in range(2): print(classify("請求書の金額が違う", ("請求", "障害", "その他")))
print(get_generator.cache_info())
