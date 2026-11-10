# 13 章 03 節：MCP サーバ（13_04）を stdio で起動し、クライアント（13_05）からツール一覧の取得と呼び出しを行う
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from blocks import block
import os
work = pathlib.Path(__file__).resolve().parent / "work"; work.mkdir(exist_ok=True)
# サーバ：db・default_since・render_order だけスタブを前に足す
(work / "order_server.py").write_text('# --- 確認用のスタブ（本文の外） ---\nclass _DB:\n    def query(self, cid, since): return [{"order_id": "O-1", "customer_id": cid, "since": since}]\ndb = _DB()\ndefault_since = lambda: "2026-08-06"\nrender_order = lambda oid: f"注文 {oid}：3,000 円"\n' + block("13", "MCPServer("), encoding="utf-8")
os.chdir(work)
os.environ["PATH"] = str(pathlib.Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]   # "python" を同じ環境にする
env = {}
exec(block("13", "stdio_client(StdioServerParameters("), env)
schema, result = env["schema"], env["result"]
print("ツール:", [(s["name"], s["description"].split("\n")[0]) for s in schema])
print("入力スキーマの引数の説明:", {k: v.get("description") for k, v in schema[0]["input_schema"]["properties"].items()})
print("呼び出し結果:", result.content[0].text.replace("\n", " ") if result.content else result)
