# 7 章 03 節：build_example・検証器・rejection_sample・audit をつなぎ、実トークナイザとスタブのモデルで動かす
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from blocks import block
for key in ("def build_example", "def verify_math", "def rejection_sample", "def audit"):
    exec(block("07", key))

# ---- 確認用：実トークナイザ・スタブのモデル・数学の検証器 ----
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
conv = [{"role": "system", "content": "簡潔に答える。"},
        {"role": "user", "content": "1+1は？"},
        {"role": "assistant", "content": "\\boxed{2}"},
        {"role": "user", "content": "2+3は？"},
        {"role": "assistant", "content": "\\boxed{5}"}]
ex = build_example(tok, conv)
spans, cur = [], []
for i, l in zip(ex["input_ids"], ex["labels"]):
    if l != IGNORE: cur.append(i)
    elif cur: spans.append(cur); cur = []
if cur: spans.append(cur)
print("トークン数", len(ex["input_ids"]), " 学習対象の割合", round(label_ratio(ex), 3))
print("全体に system 文が何回入るか:", tok.decode(ex["input_ids"]).count("<|im_start|>system"))
for s in spans: print("学習する範囲:", repr(tok.decode(s)))

import random
class StubModel:                      # 正解・誤答・短すぎる答えを混ぜて返す
    def generate(self, prompt, n, temperature):
        a = int(prompt.split("+")[0]) + int(prompt.split("+")[1].rstrip("は？"))
        r = random.Random(prompt)
        outs = []
        for _ in range(n):
            k = r.random()
            if k < 0.4:   outs.append(f"左の数に右の数を一つずつ順に足していくと {a} になる。したがって答えは \\boxed{{{a}}} である。")
            elif k < 0.6: outs.append(f"一の位から計算し、繰り上がりがあれば十の位に足す。こうして求めた答えは \\boxed{{{a}}} となる。")
            elif k < 0.8: outs.append(f"足し算をすると答えは \\boxed{{{a+1}}} になると思われる。")
            else:         outs.append(f"\\boxed{{{a}}}")
        return outs
items = [{"prompt": f"{i}+{j}は？", "answer": str(i + j)} for i, j in [(1, 1), (2, 3), (10, 7), (4, 4)]]
kept, stats = rejection_sample(StubModel(), items, verify_math, n=32)
print(stats)
audit(kept, stats, items)
print('--- 全問不合格のとき ---')
k0, s0 = rejection_sample(StubModel(), items, lambda i, x: False, n=4)
audit(k0, s0, items)
