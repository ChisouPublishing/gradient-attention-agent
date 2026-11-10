# 14 章の tail sampling の設定を、実際の OpenTelemetry Collector（Docker）で動かす。
# 本文の計装コードから OTLP で 4 種類のトレースを送り、Collector が残したものを数える。
#   失敗（ツールが ERROR）1 件・高コスト（出力 2,400 トークン）1 件・遅い（6 秒）1 件・
#   状態なしの失敗（error.type はあるが set_status なし）5 件・ふつう 200 件
# 期待：失敗・高コスト・遅いは必ず残り、状態なしの失敗とふつうは 5% 前後だけ残る。
import sys, pathlib, subprocess, json, time, types, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from blocks import block
import yaml

IMAGE = "otel/opentelemetry-collector-contrib:0.164.0"
HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "work" / "collector"; WORK.mkdir(parents=True, exist_ok=True)
(WORK / "kept.jsonl").unlink(missing_ok=True)

# 本文の YAML（processors だけ）に、受け口・出口・パイプラインを足して Collector の設定にする
cfg = yaml.safe_load(block("14", "tail_sampling"))
cfg["receivers"] = {"otlp": {"protocols": {"http": {"endpoint": "0.0.0.0:4318"}}}}
cfg["exporters"] = {"file": {"path": "/out/kept.jsonl"}}
cfg["service"] = {"pipelines": {"traces": {"receivers": ["otlp"], "processors": ["tail_sampling"], "exporters": ["file"]}}}
(WORK / "collector.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

ver = subprocess.run(["docker", "run", "--rm", IMAGE, "--version"], capture_output=True, text=True).stdout.strip()
print("Collector:", ver)
subprocess.run(["docker", "rm", "-f", "gaa-otelcol"], capture_output=True)
subprocess.run(["docker", "run", "-d", "--name", "gaa-otelcol", "-p", "4318:4318", "--user", "0",
                "-v", f"{WORK}:/out", "-v", f"{WORK / 'collector.yaml'}:/etc/otelcol-contrib/config.yaml:ro",
                IMAGE], check=True, capture_output=True)
try:
    time.sleep(3)
    logs = subprocess.run(["docker", "logs", "gaa-otelcol"], capture_output=True, text=True)
    started = "Everything is ready" in (logs.stdout + logs.stderr)
    print("本文の設定で起動したか:", started)
    if not started:
        print(logs.stdout + logs.stderr); sys.exit(1)

    # 本文の初期化をそのまま使う（送り先は既定の http://localhost:4318）
    os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    get = lambda key: block("14", key)
    exec(get("TracerProvider("))
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    mem = InMemorySpanExporter(); provider.add_span_processor(SimpleSpanProcessor(mem))  # どの trace が何かを手元で控える
    exec(get("def digest")); exec(get("def retrieve")); exec(get("def run_agent"))

    U = types.SimpleNamespace
    extract_tool_calls = lambda r: r.calls
    final_text = lambda r: "完了"
    to_messages = lambda r, calls: [{"role": "assistant", "content": "…"}]
    cross_encoder_rerank = lambda q, cands: cands
    def agent_client(out_tokens):
        it = iter([U(usage=U(input_tokens=120, output_tokens=30), stop_reason="tool_use", calls=[U(name="search_orders", args={"id": 1})]),
                   U(usage=U(input_tokens=200, output_tokens=out_tokens), stop_reason="end_turn", calls=[])])
        return U(messages=U(create=lambda **kw: next(it)))

    kind = {}
    def tid_of_last_root():
        roots = [s for s in mem.get_finished_spans() if s.parent is None]
        return format(roots[-1].context.trace_id, "032x")

    # 失敗：ツールが失敗する（出力は少ない）
    client = agent_client(300); execute = lambda c: (False, "TimeoutError: 在庫 API が 10 秒で応答しない")
    kind[run_agent("注文を調べて")["trace_id"]] = "失敗"
    # 高コスト：ツールは成功、最後の応答が 2,400 トークン
    client = agent_client(2400); execute = lambda c: (True, "ok")
    kind[run_agent("注文を調べて")["trace_id"]] = "高コスト"
    # 遅い：検索が 6 秒かかる
    vector_search = lambda q, k: (time.sleep(6), [U(id=f"d{i}", score=1 - i / 10) for i in range(k)])[1]
    retrieve("返金の手順"); kind[tid_of_last_root()] = "遅い"
    # 状態なしの失敗：error.type の属性はあるが set_status を呼ばない（本文の「★」を書き忘れた場合）
    for i in range(5):
        with tracer.start_as_current_span("execute_tool") as ts:
            ts.set_attribute("tool.ok", False); ts.set_attribute("error.type", "TimeoutError")
        kind[tid_of_last_root()] = "状態なしの失敗"
    # ふつう：速く、失敗せず、出力も少ない
    vector_search = lambda q, k: [U(id=f"d{i}", score=1 - i / 10) for i in range(k)]
    for i in range(200):
        retrieve(f"返金の手順 {i}"); kind[tid_of_last_root()] = "ふつう"
    provider.force_flush()

    # decision_wait（10 秒）の後に判定される。余裕を見て待つ
    time.sleep(16)
    subprocess.run(["docker", "stop", "-t", "5", "gaa-otelcol"], capture_output=True)  # 止めるとファイルに書き切る
    kept = set()
    for line in (WORK / "kept.jsonl").read_text().splitlines():
        for rs in json.loads(line)["resourceSpans"]:
            for ss in rs["scopeSpans"]:
                for sp in ss["spans"]:
                    kept.add(sp["traceId"].lower())
    sent = {}
    for t, k in kind.items(): sent[k] = sent.get(k, 0) + 1
    got = {}
    for t in kept: got[kind.get(t, "不明")] = got.get(kind.get(t, "不明"), 0) + 1
    for k in ["失敗", "高コスト", "遅い", "状態なしの失敗", "ふつう"]:
        print(f"{k}：送った {sent.get(k, 0)} 件 → 残った {got.get(k, 0)} 件")
    if "不明" in got: print("不明:", got["不明"])
finally:
    subprocess.run(["docker", "rm", "-f", "gaa-otelcol"], capture_output=True)
