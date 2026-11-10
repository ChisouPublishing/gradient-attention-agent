# 8 章のサンプラー・generate・best_of_n を、実モデル（Qwen2.5-0.5B-Instruct）と 5 章の説明用ロジットで確かめる
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from blocks import block
import re, pathlib, torch, numpy as np, sys
get = lambda key: block("08", key)
for key in ("def apply_repetition_penalty", "def generate(model, ids", "def best_of_n"):
    exec(get(key))

# 1) 04 節の説明用ロジット（平坦な文脈・40 語）で、生き残る候補の数が demo05 と一致するか
z = torch.tensor([[3.0 - 0.075 * i for i in range(40)]])
def survivors(**kw):
    torch.manual_seed(0)
    # sample() の切り詰めだけを見るため、抽選前の確率を取り出す
    lg = z.clone()
    if kw.get("temperature", 1.0) != 1.0: lg = lg / kw["temperature"]
    srt, idx = lg.sort(descending=True, dim=-1); cum = srt.softmax(-1).cumsum(-1)
    drop = cum - srt.softmax(-1) >= kw["top_p"]
    return int((~drop).sum())
print("平坦・top-p=0.9 の生き残り:", survivors(top_p=0.9), "／ T=1.5:", survivors(temperature=1.5, top_p=0.9))
torch.manual_seed(0)
draws = torch.cat([sample(z, temperature=1.0, top_p=0.9) for _ in range(2000)]).flatten()
print("2,000 回引いた最大の順位（0 始まり）:", int(draws.max()), "（26 語なら 25 以下）")
print("T=0 の形:", tuple(sample(z.repeat(3, 1), temperature=0).shape))
# 反復ペナルティ：正のロジットは割られ、負のロジットは掛けられる
lg = torch.tensor([[2.0, -2.0, 1.0]])
print("反復ペナルティ 2.0 を語 0・1 に:", apply_repetition_penalty(lg, torch.tensor([[0, 1]]), 2.0).tolist())

# 2) 実モデルで generate と best_of_n
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name); model = AutoModelForCausalLM.from_pretrained(name).eval()
ids = tok("日本の首都は", return_tensors="pt").input_ids
torch.manual_seed(0)
print("generate T=0:", repr(tok.decode(generate(model, ids, max_new=8, temperature=0)[0][ids.size(1):])))
torch.manual_seed(0)
best, all_ = best_of_n(model, tok, ids, lambda xs: [-abs(len(x) - 10) for x in xs], n=4, max_new=12)
for x, s in all_: print(f"  候補 score={s:3d} {x!r}")
print("best_of_n の選択:", repr(best))
torch.manual_seed(0)
_, zero = best_of_n(model, tok, ids, lambda xs: [0] * len(xs), n=4, max_new=12, temperature=0)
print("temperature=0 の n 本が全部同じか:", len({x for x, _ in zero}) == 1)
