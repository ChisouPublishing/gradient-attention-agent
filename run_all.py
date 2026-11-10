#!/usr/bin/env python3
"""本のコードをすべて実行し、出力を chNN/outputs/ に保存する。

  python run_all.py              # すべて
  python run_all.py --only 08 12 # 章を選ぶ
  python run_all.py --fast       # モデルの取得や長い計算が要るものを飛ばす
  python run_all.py --name=vllm  # 名前の先頭で選ぶ
  （8 章の vLLM と 14 章の Collector は Docker で動かす。Docker が動いていなければ飛ばす）

「ブロック」は個別.ipynbの本文対応セルから取り出し、別Pythonプロセスで実行する。
「run_」は本文の断片をつないで動かす検証スクリプト（chNN/run_*.py）。
"""
import os, pathlib, shutil, subprocess, sys, time

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / "common"))
from blocks import block  # noqa: E402

# (章, 名前, ブロックを探す語 または run スクリプト名, 重いか[, 渡す引数])
RUNS = [
    ("02", "baseline",        "make_classification(n_samples", True),
    ("03", "gd1d",            "def grad(w):", False),
    ("03", "linreg",          "np.linspace(0.5, 4.0, 14)", False),
    ("03", "optimizers",      "def momentum(", False),
    ("03", "training_loop",   "LambdaLR(opt, factor)", False),
    ("03", "before_fix",      "直す前の書き方", False),
    ("03", "scheduler",       "run_scheduler.py", False),
    ("04", "backprop",        "def layer_grads(", False),
    ("05", "tokens",          "corpus = {\"low\"", True),
    ("06", "attention",       "猫 が 魚 を 食べ た", False),
    ("06", "position",        "def sinusoidal(", False),
    ("06", "matrix_kvcache",  "run_matrix_and_kvcache.py", False),
    ("07", "grpo",            "def grpo_advantages(", False),
    ("07", "sft_pipeline",    "run_sft_pipeline.py", True),
    ("07", "prompt",          "SHOTS_BAL", True),
    ("08", "sampling",        "SHARP = {", False),
    ("08", "sampler",         "run_sampler.py", True),
    ("08", "outlines",        "run_outlines.py", True),
    ("08", "constrained",     "run_constrained.py", True),
    ("08", "cot_0.5B",        "④ ③を 5 本引いて多数決", True, ["Qwen/Qwen2.5-0.5B-Instruct"]),
    ("08", "cot_1.5B",        "④ ③を 5 本引いて多数決", True, ["Qwen/Qwen2.5-1.5B-Instruct"]),
    ("08", "vllm_0.5B",       "run_vllm.sh", True, ["Qwen/Qwen2.5-0.5B-Instruct"]),   # Docker（vLLM の CPU 版）
    ("08", "vllm_1.5B",       "run_vllm.sh", True, ["Qwen/Qwen2.5-1.5B-Instruct"]),
    ("08", "vllm_0.5B_allow_ws", "run_vllm.sh", True, ["Qwen/Qwen2.5-0.5B-Instruct", "allow_ws"]),
    ("08", "vllm_1.5B_allow_ws", "run_vllm.sh", True, ["Qwen/Qwen2.5-1.5B-Instruct", "allow_ws"]),
    ("08", "tot_got",         "def tot(nums, beam)", False),
    ("09", "online_softmax",  "最大値が後ろに来る", False),
    ("09", "exactness",       "run_exactness.py", False),
    ("11", "bandwidth",       "np.copyto(b, a)", True),
    ("12", "rag",             "run_rag.py", True),
    ("12", "ann",             "IvfFlat(", True),
    ("13", "agent",           "run_agent.py", False),
    ("13", "mcp",             "run_mcp.py", False),
    ("13", "memory",          "class Memory:", True),
    ("13", "function_calling", "def tool_calls(text)", True),
    ("14", "eval_statistics",  "def mcnemar_exact", False),
    ("14", "otel",            "run_otel.py", False),
    ("14", "collector",       "run_collector.py", True),   # Docker（OpenTelemetry Collector）
    ("18", "tensor_parallel", "A_parts = np.split", False),
    ("18", "ssm",             "def recurrent(", False),
    ("20", "tensor",          "torch.from_numpy(a)", False),
    ("20", "data",            "class Squares(Dataset)", True),
    ("20", "model",           "class MLP(nn.Module)", False),
    ("20", "autograd",        "detach したものは記録されるか", False),
    ("20", "train",           "def train_one_epoch", True),
    ("20", "save",            "whole.pt", False),
]

def main():
    args = sys.argv[1:]
    only = [a for a in args if a.isdigit()]
    names = [a.split("=", 1)[1] for a in args if a.startswith("--name=")]   # 名前の先頭で選ぶ（例：--name=vllm）
    fast = "--fast" in args
    env = dict(os.environ, PYTHONIOENCODING="utf-8",
               PATH=str(pathlib.Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    results = []
    DOCKER = {"run_vllm.sh", "run_collector.py"}
    docker = None
    for ch, name, target, heavy, *extra in RUNS:
        if (only and ch not in only) or (fast and heavy) or (names and not any(name.startswith(n) for n in names)):
            continue
        if target in DOCKER:
            if docker is None:
                docker = False
                if "--skip-docker" not in args and shutil.which("docker"):
                    try:
                        docker = subprocess.run(["docker", "info"], capture_output=True,
                                                timeout=10).returncode == 0
                    except (OSError, subprocess.TimeoutExpired):
                        pass
            if not docker:
                print(f"--  {ch} {name:16s} Docker を利用しないので飛ばす", flush=True)
                continue
        d = CODE / f"ch{ch}"
        work, outs = d / "work", d / "outputs"
        work.mkdir(exist_ok=True); outs.mkdir(exist_ok=True)
        if target.startswith("run_"):
            script = d / target
        else:                                       # 本文のブロックをそのまま実行する
            script = work / f"block_{name}.py"
            script.write_text(block(ch, target), encoding="utf-8")
        t0 = time.time()
        cmd = ["sh", str(script)] if target.endswith(".sh") else [sys.executable, str(script)]
        r = subprocess.run([*cmd, *(extra[0] if extra else [])],
                           cwd=work, env=env, capture_output=True, text=True)
        (outs / f"{name}.out").write_text(r.stdout, encoding="utf-8")
        for raw in work.glob("*_raw.json"):            # 証跡のファイル（7 章の全件の応答など）は outputs に残す
            raw.replace(outs / raw.name)
        err = outs / f"{name}.err"
        if r.returncode:
            err.write_text(r.stderr, encoding="utf-8")
        elif err.exists():
            err.unlink()                              # 前回の失敗の記録を消す
        results.append((ch, name, r.returncode, time.time() - t0))
        print(f"{'OK ' if r.returncode == 0 else 'NG '} {ch} {name:16s} {time.time() - t0:6.1f} 秒", flush=True)
    ng = [x for x in results if x[2]]
    print(f"{len(results) - len(ng)}/{len(results)} 件が終了コード 0" + ("" if not ng else "。失敗は chNN/outputs/*.err"))
    return 1 if ng else 0

if __name__ == "__main__":
    sys.exit(main())
