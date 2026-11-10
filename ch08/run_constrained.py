# 8 章 07 節：自作の制約デコーディング（許可トークンのマスク）と、検証エラーを返して再試行する書き方
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from blocks import block
import re, pathlib, torch, json
get = lambda key: block("08", key)
exec(get("def build_allow_mask")); exec(get("def parse_with_retry"))
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name); model = AutoModelForCausalLM.from_pretrained(name).eval()
allowed = build_allow_mask(tok, r"\d+", model.config.vocab_size)
print("許可されたトークン数:", int(allowed.sum()), "/", allowed.numel())
ids = tok("2026 年の 3 倍は", return_tensors="pt").input_ids
torch.manual_seed(0)
out = generate_constrained(model, tok, ids, allowed, max_new=6)
gen = tok.decode(out[0, ids.size(1):])
print("生成:", repr(gen), " 数字だけか:", gen.isdigit())

# 再試行：1 回目は壊れた JSON、2 回目は型の合わない JSON、3 回目で正しい JSON を返すスタブ
from pydantic import BaseModel
class Item(BaseModel):
    name: str
    qty: int
replies = iter(['{"name": "りんご", "qty": ', '{"name": "りんご", "qty": "三"}', '{"name": "りんご", "qty": 3}'])
log = []
def call_llm(messages):
    log.append(len(messages)); return next(replies)
print("再試行の結果:", parse_with_retry(call_llm, "JSON で答えて", Item), " 各回のメッセージ数:", log)
