"""
尺寸泛化基准：把若干 checkpoint 放在若干尺寸上统一评估，贪心和采样各一遍。

  python bench.py ckpt/stage3.pt ckpt/stage4.pt ckpt/stage6.pt --sizes 5 8 12 16
  python bench.py ckpt/stage6.pt --sizes 5 8 12 16 --json results/stage6.json

输出每格是「成功率 / 步数÷最优」。算法 agent 作为参照一起跑。
"""
import argparse
import json
import os

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
    ap.add_argument("--modes", nargs="+", default=["greedy", "sample"])
    ap.add_argument("--json", default=None, help="每算完一格就写入，中途被杀也不丢")
    ap.add_argument("--no-algo", action="store_true")
    ap.add_argument("--threads", type=int, default=1, help="几个进程并行跑时设 1，避免抢核")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    results = {}
    if args.json and os.path.exists(args.json):
        results = json.load(open(args.json))

    def save(key, val, label):
        results[key] = val
        print(f"{label:18s} succ {val[0]:.2f} ratio {val[1]:.2f}", flush=True)
        if args.json:
            os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
            json.dump(results, open(args.json, "w"), indent=1)

    for path in args.ckpts:
        model, cfg = model_from_checkpoint(torch.load(path))
        for size in args.sizes:
            n = EPISODES.get(size, 20)
            for mode in args.modes:
                key = f"{path}|{size}|{mode}"
                if key in results:
                    continue
                s, r = bench(model, cfg, size, n, mode == "greedy", args.seed_offset)
                save(key, (s, r), f"{path} {size}x{size} {mode}")
    if not args.no_algo:
        for size in args.sizes:
            key = f"algo|{size}|-"
            if key in results:
                continue
            s, r = bench_algo(size, EPISODES.get(size, 20), cfg["key_door"], args.seed_offset)
            save(key, (s, r), f"algo {size}x{size}")


if __name__ == "__main__":
    main()
