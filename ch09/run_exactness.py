# 9 章：3 パス・2 パス・1 パス・タイル化を素朴実装と突き合わせる（本文のコードをそのまま使う）
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from blocks import block
import re, pathlib, numpy as np
from numpy import exp, inf, zeros
get = lambda key: block("09", key)
two_pass_src, one_pass_src = get("# 2 パス"), get("# 1 パス")
exec(get("def flash_attention"))

rng = np.random.default_rng(0)
N, d = 512, 64
Q, K, V = (rng.normal(size=(N, d)) for _ in range(3))
S = Q @ K.T / np.sqrt(d)

def naive(S, V):                                 # 素朴実装（N×N を作って softmax）
    P = np.exp(S) / np.exp(S).sum(1, keepdims=True)
    return P @ V

def three_pass(S, V):                            # 本文の safe softmax（3 パス）
    out = []
    for x in S:
        m = x.max(); l = exp(x - m).sum(); p = exp(x - m) / l
        out.append(p @ V)
    return np.array(out)

def run_loop(src, S, V):                         # 本文の 2 パス・1 パスを行ごとに実行
    src = src.replace("return o", "")
    out = []
    for row in S:
        env = {"exp": exp, "inf": inf, "zeros": zeros, "d": V.shape[1],
               "scores": row, "values": V, "max": max}
        exec(src, env); out.append(env["o"])
    return np.array(out)

ref = naive(S, V)
rows = [("3 パス safe softmax", three_pass(S, V)),
        ("2 パス オンライン正規化", run_loop(two_pass_src, S, V)),
        ("1 パス（本文のコード）", run_loop(one_pass_src, S, V)),
        ("タイル化 Br=64 / Bc=64", flash_attention(Q, K, V, 64, 64)),
        ("タイル化 Br=16 / Bc=256", flash_attention(Q, K, V, 16, 256))]
for name, o in rows:
    print(f"{name:24s} 最大絶対差 {np.abs(o - ref).max():.2e}")
O32 = flash_attention(*(a.astype(np.float32) for a in (Q, K, V)), 64, 64)
print(f"{'タイル化（fp32 で実行）':24s} 最大絶対差 {np.abs(O32 - ref).max():.2e}   fp32 の機械イプシロン {np.finfo(np.float32).eps:.2e}")
# 素朴実装が fp16 で壊れる閾値
x16 = np.array([11.0, 11.1], dtype=np.float16)
print("fp16: exp(11.0) =", np.exp(x16[0]), " exp(11.1) =", np.exp(x16[1]), " 最大値", np.finfo(np.float16).max)
