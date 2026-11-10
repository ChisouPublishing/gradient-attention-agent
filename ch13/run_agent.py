# 13 章のエージェントのコードを本文から取り出し、スタブの LLM・ツールで振る舞いを確かめる
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from blocks import block
import re, pathlib, json, types, hashlib
get = lambda key: block("13", key)
exec(get("TOOLS = ["))
src = get("def run_agent")
exec(src)
if "def err(" not in src:
    def err(call, msg): return {"type": "tool_result", "tool_use_id": call["id"], "content": msg, "is_error": True}
    def stop(reason, messages): return {"answer": None, "stopped": reason, "steps": len(messages)}
TOOL_IMPL = {"search_orders": lambda customer_id, since=None: [{"order_id": "O-1", "amount": 3000}],
             "issue_refund": lambda order_id, amount: {"ok": True}}
approved = lambda call: False

def make_llm(script):                      # 台本どおりに応答するスタブ
    it = iter(script)
    def llm(messages, tools):
        content = next(it)
        text = "".join(c.get("text", "") for c in content if c["type"] == "text")
        return types.SimpleNamespace(content=content, text=text, cost=0.01)
    return llm
call = lambda i, name, **a: {"type": "tool_use", "id": f"t{i}", "name": name, "input": a}
# 1) 1 回検索して答える
llm = make_llm([[call(1, "search_orders", customer_id="CUS-00123")], [{"type": "text", "text": "注文は O-1 です"}]])
print("通常:", run_agent("注文を調べて"))
# 2) 同じ呼び出しを繰り返す → ②で止められ、③ステップ上限で打ち切り
llm = make_llm([[call(i, "search_orders", customer_id="CUS-00123")] for i in range(20)])
r = run_agent("注文を調べて", max_steps=4)
print("堂々巡り:", r["stopped"], "／ 2 回目の結果:", r and "繰り返しています" in json.dumps(r, ensure_ascii=False) or "（下で確認）")
msgs_seen = []
llm0 = make_llm([[call(i, "search_orders", customer_id="CUS-00123")] for i in range(20)])
def llm(messages, tools): msgs_seen.append(json.dumps(messages[-1], ensure_ascii=False)); return llm0(messages, tools)
run_agent("注文を調べて", max_steps=3)
print("  2 回目に返したツール結果に警告が入るか:", "繰り返しています" in msgs_seen[2])
# 3) 承認の要る返金 → 承認待ちのエラーを返す
llm = make_llm([[call(1, "issue_refund", order_id="O-1", amount=3000)], [{"type": "text", "text": "承認をお願いします"}]])
print("承認なしの返金:", run_agent("返金して")["answer"])
# 4) コスト上限
llm = make_llm([[call(i, "search_orders", customer_id=f"C{i}")] for i in range(20)])
print("コスト上限:", run_agent("調べて", budget_usd=0.025)["stopped"])

# 並列実行（読み取りだけを並列に）
calls = [call(1, "search_orders", customer_id="A"), call(2, "issue_refund", order_id="O", amount=1), call(3, "search_orders", customer_id="B")]
exec(get("ThreadPoolExecutor"))
print("並列実行の結果の順:", [r["tool_use_id"] for r in results])

# 致命的な組み合わせ
class Agent:
    def __init__(self, names): self.names = names
blk = get("class Cap(Flag)")
exec(blk.split("# OK :")[0])
print("OK の組み合わせ:", build_agent(["fetch_url", "calculator"]).names)
try:
    build_agent(["fetch_url", "search_customer_db", "send_email"])
except ValueError as e:
    print("NG の組み合わせ: ValueError", str(e)[:30])
print("read_email + search_customer_db + calculator:", end=" ")
try: build_agent(["read_email", "search_customer_db", "calculator"]); print("起動する")
except ValueError: print("止まる")

# 承認を引数に結びつける
exec(get("def execute(call, *, approvals)"))
hash_args = lambda a: hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()
c = call(9, "issue_refund", order_id="O-1", amount=3000)
tok = types.SimpleNamespace(args_hash=hash_args({"order_id": "O-1", "amount": 3000}))
print("承認どおり:", execute(c, approvals={"t9": tok}))
c2 = call(9, "issue_refund", order_id="O-1", amount=30000)
print("承認後に金額を変えた:", execute(c2, approvals={"t9": tok})["content"][:20])
