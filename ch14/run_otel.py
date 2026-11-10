# 14 章の計装のコードを本文から取り出し、span をメモリに記録して中身を確かめる
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from blocks import block
import re, pathlib, types, yaml
get = lambda key: block("14", key)
setup = get("TracerProvider(")
exec(setup)                                             # 本文の初期化（OTLP 送信先は未起動でも作れる）
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
mem = InMemorySpanExporter(); provider.add_span_processor(SimpleSpanProcessor(mem))
exec(get("def digest"))
# スタブ：LLM クライアント・検索・リランカー・ツール
U = types.SimpleNamespace
script = iter([U(usage=U(input_tokens=120, output_tokens=30), stop_reason="tool_use", calls=[U(name="search_orders", args={"id": 1})]),
               U(usage=U(input_tokens=200, output_tokens=2400), stop_reason="end_turn", calls=[])])
client = U(messages=U(create=lambda **kw: next(script)))
vector_search = lambda q, k: [U(id=f"d{i}", score=1 - i / 10) for i in range(k)]
cross_encoder_rerank = lambda q, cands: sorted(cands, key=lambda c: int(c.id[1:]) % 7)
extract_tool_calls = lambda r: r.calls
final_text = lambda r: "完了"
to_messages = lambda r, calls: [{"role": "assistant", "content": "…"}]
execute = lambda c: (False, "TimeoutError: 在庫 API が 10 秒で応答しない")
exec(get("def retrieve")); exec(get("def run_agent"))
ranked = retrieve("返金の手順")
out = run_agent("注文を調べて")
for s in mem.get_finished_spans():
    a = dict(s.attributes)
    keep = {k: v for k, v in a.items() if k in ("gen_ai.usage.output_tokens", "rag.rank_before", "tool.ok", "error.type", "agent.stop_reason")}
    print(f"{s.name:14s} status={s.status.status_code.name:5s} {keep}")
print("応答の trace_id:", out["trace_id"][:8], "… 長さ", len(out["trace_id"]))
cfg = yaml.safe_load(block("14", "tail_sampling"))
print("YAML のポリシー:", [(p["name"], p["type"]) for p in cfg["processors"]["tail_sampling"]["policies"]])

# LLM 呼び出しが例外で終わったとき、chat span の状態が ERROR になるか
mem.clear()
def boom(**kw): raise TimeoutError("応答なし")
client = U(messages=U(create=boom))
try: call_llm([{"role": "user", "content": "x"}])
except TimeoutError: pass
s = mem.get_finished_spans()[0]
print("例外時の chat:", s.status.status_code.name, dict(s.attributes).get("error.type"), "例外イベント", len(s.events))
