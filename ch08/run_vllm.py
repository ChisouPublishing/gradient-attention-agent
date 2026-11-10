# 8 章の vLLM のブロックを、vLLM の CPU 版（Linux arm64 の Docker イメージ）で実行する。
# 本文の Qwen/Qwen3-8B は CPU では重いので、モデル名だけ引数のモデルに差し替える（ほかは本文のまま）。
# このスクリプトはコンテナの中で動く。Mac からは run_vllm.sh で起動する。
import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from blocks import block
get = lambda key: block("08", key)
name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
# 2 つ目の引数に allow_ws を付けると、本文の disable_any_whitespace を外す（JSON の空白を許す。vLLM の既定）。
# なお 0.33.0 では、リクエストごとの StructuredOutputsParams(disable_any_whitespace=True) はどこからも読まれず、効かない
allow_ws = len(sys.argv) > 2 and sys.argv[2] == "allow_ws"

def main():
    # 本文の Ticket の定義（Outlines のブロックの前半。Outlines の部分は使わない）
    g = {}
    exec(get("outlines.from_transformers").split("# --- Outlines")[0], g); Ticket = g["Ticket"]

    import vllm
    print("vLLM:", vllm.__version__, "／モデル:", name)
    src = get("from vllm import LLM")
    assert 'model="Qwen/Qwen3-8B"' in src
    src = src.replace('model="Qwen/Qwen3-8B"', f'model="{name}"')
    if allow_ws:
        old = '"disable_any_whitespace": True'
        assert old in src
        src = src.replace(old, '"disable_any_whitespace": False')
    print("エンジンの設定:", "空白を許す（vLLM の既定）" if allow_ws else "本文のまま（空白を禁じる）")
    prompts = ["次の問い合わせを分類して JSON で答えてください：ログイン画面で 500 エラーが出ます。",
               "次の問い合わせを分類して JSON で答えてください：請求書の金額が先月と違います。",
               "次の問い合わせを分類して JSON で答えてください：CSV で書き出す機能が欲しいです。"]
    prompt = prompts[0]
    g.update(prompt=prompt, Ticket=Ticket)
    t = time.time(); exec(src, g); dt = time.time() - t           # 本文のブロック（モデルの読み込みと 1 回目の生成）
    print(f"本文のブロック：{dt:.0f} 秒（モデルの読み込みを含む）")

    # 同じ llm と SamplingParams で 3 つの問い合わせを試し、型として通るかを確かめる
    llm, SamplingParams, StructuredOutputsParams = g["llm"], g["SamplingParams"], g["StructuredOutputsParams"]
    print("本文のブロックの出力:", g["out"][0].outputs[0].text[:120].replace("\n", " "))
    schema = Ticket.model_json_schema()
    conds = [("構造化出力", lambda: StructuredOutputsParams(json=schema))]
    if allow_ws:
        conds.append(("制約なし", lambda: None))
    raw = []
    for label, so in conds:
        ok, length, n_ws = 0, 0, 0
        for p in prompts:
            for seed in range(3):                       # 温度 0.2 なので、種を変えて 3 回ずつ
                sp = SamplingParams(temperature=0.2, max_tokens=512, seed=seed, structured_outputs=so())
                o = llm.generate(p, sp, use_tqdm=False)[0].outputs[0]
                text, fin = o.text, o.finish_reason
                try:
                    Ticket.model_validate_json(text); ok += 1; res = "通った"
                except Exception:
                    res = "通らない"
                    if fin == "length":
                        length += 1
                        tail = text[-200:]
                        if tail.strip() == "" or sum(ch.isspace() for ch in tail) > 150: n_ws += 1   # 空白だけが続いて打ち切られた
                raw.append({"cond": label, "prompt": p, "seed": seed, "result": res, "finish": fin,
                            "n_tokens": len(o.token_ids), "text": text})
        n = len(prompts) * 3
        print(f"{label}：型として通った {ok}/{n}（打ち切り {length} 件、うち空白が続いたもの {n_ws} 件）")
    tag = name.split("-")[-2] + ("_allow_ws" if allow_ws else "")
    pathlib.Path(f"ch08/outputs/vllm_{tag}_raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1))


# vLLM はエンジンを別プロセス（spawn）で起動する。スクリプトの最上位で LLM を作ると子プロセスが失敗するので、main の中で作る
if __name__ == "__main__":
    main()
