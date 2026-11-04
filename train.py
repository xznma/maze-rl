"""
训练主循环 + 课程阶段。

  python train.py --stage 0                    # 固定 3x3，无钥匙门（验证代码）
  python train.py --stage 1                    # 固定 5x5，无钥匙门
  python train.py --stage 2                    # 随机 5x5，无钥匙门
  python train.py --stage 3 --init ckpt/stage2.pt   # 随机 5x5，有钥匙门，接着练
  python train.py --stage 4 --init ckpt/stage3.pt   # 随机 8x8，有钥匙门
  也可以用 --h --w --random --key-door 自定义。
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from env import VecEnv
from model import ActorCritic
from ppo import collect, compute_gae, update

STAGES = {
    0: dict(h=3, w=3, random=False, key_door=False, total_steps=100_000, target=0.98),
    1: dict(h=5, w=5, random=False, key_door=False, total_steps=400_000, target=0.98),
    2: dict(h=5, w=5, random=True, key_door=False, total_steps=3_000_000, target=0.95),
    3: dict(h=5, w=5, random=True, key_door=True, total_steps=6_000_000, target=0.99),
    4: dict(h=8, w=8, random=True, key_door=True, total_steps=4_000_000, target=0.95),
    # 阶段 5：换成"已探索地图"观测，混合尺寸 5–10 训练，目标是泛化到更大的迷宫
    5: dict(h=8, w=8, random=True, key_door=True, obs="map", size_range=(5, 10),
            total_steps=6_000_000, target=0.95),
}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=None)
    ap.add_argument("--h", type=int); ap.add_argument("--w", type=int)
    ap.add_argument("--random", action="store_true", help="每局换迷宫")
    ap.add_argument("--key-door", action="store_true")
    ap.add_argument("--fixed-seed", type=int, default=0, help="非 random 时用的迷宫 seed")
    ap.add_argument("--obs", choices=["local", "map"], default=None)
    ap.add_argument("--map-radius", type=int, default=12)
    ap.add_argument("--size-range", type=int, nargs=2, default=None,
                    help="每局随机抽尺寸，如 --size-range 5 10")
    ap.add_argument("--total-steps", type=int)
    ap.add_argument("--target", type=float, help="最近 100 局成功率达到即提前停止")
    ap.add_argument("--init", type=str, help="从 checkpoint 继续训练")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.stage is not None:
        for k, v in STAGES[args.stage].items():
            if getattr(args, k, None) in (None, False):
                setattr(args, k, v)
    if args.obs is None:
        args.obs = "local"
    if args.size_range is not None:
        args.size_range = tuple(args.size_range)
    if args.out is None:
        args.out = f"ckpt/stage{args.stage}.pt" if args.stage is not None else "ckpt/custom.pt"
    return args


def main():
    args = parse()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    vec = VecEnv(args.num_envs, h=args.h, w=args.w, with_key_door=args.key_door,
                 fixed_seed=None if args.random else args.fixed_seed,
                 obs_mode=args.obs, map_radius=args.map_radius, size_range=args.size_range)
    model = ActorCritic(args.obs, args.map_radius)
    if args.init:
        model.load_state_dict(torch.load(args.init)["model"])
        print(f"loaded {args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    obs = vec.reset()
    state = model.initial_state(args.num_envs)
    first = torch.ones(args.num_envs)
    recent = deque(maxlen=100)
    steps, t0 = 0, time.time()
    steps_per_iter = args.num_envs * args.rollout
    n_iters = args.total_steps // steps_per_iter
    size = f"{args.size_range[0]}–{args.size_range[1]}" if args.size_range else f"{args.h}x{args.w}"
    print(f"maze {size} random={args.random} key_door={args.key_door} obs={args.obs} "
          f"| {n_iters} iters × {steps_per_iter} steps "
          f"| {sum(p.numel() for p in model.parameters()):,} params")

    for it in range(1, n_iters + 1):
        frac = 1.0 - (it - 1) / n_iters
        for g in opt.param_groups:
            g["lr"] = args.lr * frac
        buf, h0, obs, state, first, last_v, infos = collect(
            model, vec, obs, state, first, args.rollout, args.gamma)
        compute_gae(buf, last_v, args.gamma, args.lam)
        stats = update(model, opt, buf, h0, args.epochs, args.minibatches, args.clip,
                       args.vf_coef, args.ent_coef, 0.5)
        steps += steps_per_iter
        recent.extend(infos)

        if it % 10 == 0 or it == n_iters:
            succ = np.mean([i["success"] for i in recent]) if recent else 0.0
            length = np.mean([i["steps"] for i in recent]) if recent else 0.0
            print(f"it {it:5d} | steps {steps:8d} | succ {succ:5.2f} | len {length:6.1f} "
                  f"| ent {stats['entropy']:.3f} | vloss {stats['value_loss']:.4f} "
                  f"| kl {stats['approx_kl']:.4f} | {steps / (time.time() - t0):.0f} sps")
            if len(recent) >= 100 and args.target and succ >= args.target:
                print(f"reached target success {args.target} — stopping early")
                break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
