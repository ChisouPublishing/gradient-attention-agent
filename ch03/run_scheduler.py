# 3 章 11 節：学習ループ（03_04）に続けて、SequentialLR の書き方（03_05）を実行し、コメントの学習率を確かめる
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
from blocks import block
env = {}
exec(block("03", "LambdaLR(opt, factor)"), env)        # 学習ループを 1 回回す（03_04）
exec(block("03", "SequentialLR("), env)                      # スケジューラを差し替える
sch, opt, warmup, total = env["sch"], env["opt"], env["warmup"], env["total_steps"]
print("warmup", warmup, "total", total)
for s in range(total):
    if s in (0, warmup, total - 1):
        print("step", s + 1, f"{sch.get_last_lr()[0]:.5f}")
    opt.step(); sch.step()
