# 6 章 03 節：行列版の attention() が 1 語ずつの計算（06_01）と一致するか。マルチヘッドの step()（KV キャッシュ）が一括計算と一致するか
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from blocks import block
import numpy as np, torch
d4 = {}; exec(block("06", "猫 が 魚 を 食べ た"), d4)            # 1 語ずつの計算（表示も走る）
mx = {}; exec(block("06", "def attention(Q, K, V, mask=None):"), mx)
_, W = mx["attention"](d4["Q"], d4["K"], d4["V"], mx["causal"])
diff = max(abs(W[i] - d4["attention"](d4["Q"][i], d4["K"], d4["V"], d4["causal"][i])[1]).max() for i in range(6))
print("行列版と 1 語ずつの計算の最大差:", diff)

env = {}
exec(block("06", "class MultiHeadSelfAttention"), env)
exec(block("06", "def step(self, x_new, cache)"), env)
M = env["MultiHeadSelfAttention"]; M.step = env["step"]
torch.manual_seed(0)
B, T, C, H = 2, 7, 32, 4
m = M(C, H).eval()
x = torch.randn(B, T, C)
with torch.no_grad():
    full = m(x)
cache, outs = {"k": None, "v": None}, []
for t in range(T):
    o, cache = m.step(x[:, t:t+1], cache); outs.append(o)
print("一度に計算 vs KV キャッシュで逐次：最大差", (full - torch.cat(outs, dim=1)).abs().max().item())
print("キャッシュの形", tuple(cache["k"].shape), "（B, h, T, dk）")
