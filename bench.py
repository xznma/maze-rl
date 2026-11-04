"""
尺寸泛化基准：把若干 checkpoint 放在若干尺寸上统一评估，贪心和采样各一遍。

  python bench.py ckpt/stage3.pt ckpt/stage4.pt ckpt/stage6.pt --sizes 5 8 12 16
  python bench.py ckpt/stage6.pt --sizes 5 8 12 16 --json results/stage6.json

输出每格是「成功率 / 步数÷最优」。算法 agent 作为参照一起跑。
"""
import argparse
import json

import numpy as np
import torch

from env import MazeRLEnv, optimal_steps
from eval import run_algo, run_rl
from model import model_from_checkpoint

EPISODES = {5: 100, 8: 100, 12: 50, 16: 30, 20: 20}


def bench(model, cfg, size, n, greedy, seed_offset):
    env = MazeRLEnv(size, size, cfg["key_door"], obs_mode=model.obs_mode, map_radius=model.R)
    succ, ratio = [], []
    for ep in range(n):
        ok, steps, _ = run_rl(model, env, seed_offset + ep, greedy=greedy)
        succ.append(ok)
        if ok:
            ratio.append(steps / optimal_steps(env.grid, (1, 1)))
    return float(np.mean(succ)), float(np.mean(ratio)) if ratio else float("nan")


def bench_algo(size, n, key_door, seed_offset):
    env = MazeRLEnv(size, size, key_door)
    succ, ratio = [], []
    for ep in range(n):
        env.reset(seed_offset + ep)
        ok, steps = run_algo(size, size, key_door, seed_offset + ep, env.max_steps)
        succ.append(ok)
        if ok:
            ratio.append(steps / optimal_steps(env.grid, (1, 1)))
    return float(np.mean(succ)), float(np.mean(ratio))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--sizes", type=int, nargs="+", default=[5, 8, 12, 16])
    ap.add_argument("--seed-offset", type=int, default=1_000_000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-algo", action="store_true")
    args = ap.parse_args()

    results = {}
    for path in args.ckpts:
        model, cfg = model_from_checkpoint(torch.load(path))
        for size in args.sizes:
            n = EPISODES.get(size, 20)
            for mode, greedy in (("greedy", True), ("sample", False)):
                s, r = bench(model, cfg, size, n, greedy, args.seed_offset)
                results[f"{path}|{size}|{mode}"] = (s, r)
                print(f"{path:18s} {size:2d}x{size:<2d} {mode:6s} succ {s:.2f} ratio {r:.2f}", flush=True)
    if not args.no_algo:
        key_door = cfg["key_door"]
        for size in args.sizes:
            s, r = bench_algo(size, EPISODES.get(size, 20), key_door, args.seed_offset)
            results[f"algo|{size}|-"] = (s, r)
            print(f"{'algo':18s} {size:2d}x{size:<2d} {'-':6s} succ {s:.2f} ratio {r:.2f}", flush=True)
    if args.json:
        import os
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
