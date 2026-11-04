"""
评估：在没见过的迷宫上跑 N 局，和算法 agent (ExplorerAgent) 对比。

  python eval.py ckpt/stage2.pt --episodes 100
  python eval.py ckpt/stage2.pt --episodes 100 --render 3     # 打印前 3 局轨迹
  python eval.py ckpt/stage1.pt --fixed-seed 0                 # 固定迷宫阶段用
"""
import argparse

import numpy as np
import torch

from env import MazeRLEnv, optimal_steps
from maze_game import ExplorerAgent, MazeEnv, build_level, render
from model import model_from_checkpoint


def run_rl(model, env, seed, greedy=True):
    obs = env.reset(seed)
    state = model.initial_state(1)
    first = torch.ones(1)
    visited = {env.pos}
    done, info = False, {}
    with torch.no_grad():
        while not done:
            logits, _, state = model(torch.as_tensor(obs)[None, None], state, first[None])
            a = logits[0, 0].argmax() if greedy else torch.distributions.Categorical(
                logits=logits[0, 0]).sample()
            obs, _, done, info = env.step(int(a))
            visited.add(env.pos)
            first = torch.zeros(1)
    return info["success"], info["steps"], visited


def run_algo(h, w, key_door, seed, max_steps):
    grid, start = build_level(h, w, seed, key_door)
    env, agent = MazeEnv(grid, start), ExplorerAgent()
    t = 0
    while not env.done and t < max_steps:
        env.step(agent.act(env.pos, env.observe(), env.has_key))
        t += 1
    return env.done, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--fixed-seed", type=int, default=None)
    ap.add_argument("--seed-offset", type=int, default=1_000_000, help="避开训练用过的 seed")
    ap.add_argument("--render", type=int, default=1)
    ap.add_argument("--greedy", action="store_true",
                    help="每步取 argmax。默认按策略采样：贪心是确定性的，容易在大迷宫里陷入循环，"
                         "成功率会明显低估（16x16 上 23%% vs 63%%）")
    ap.add_argument("--h", type=int); ap.add_argument("--w", type=int)
    args = ap.parse_args()

    model, cfg = model_from_checkpoint(torch.load(args.ckpt))
    if args.h: cfg["h"] = args.h          # 可以在比训练时更大的迷宫上零样本测试
    if args.w: cfg["w"] = args.w
    env = MazeRLEnv(cfg["h"], cfg["w"], cfg["key_door"],
                    obs_mode=model.obs_mode, map_radius=model.R)
    print(f"maze {cfg['h']}x{cfg['w']} key_door={cfg['key_door']} obs={model.obs_mode} "
          f"| {args.episodes} episodes")

    rl_succ, rl_ratio, al_succ, al_ratio = [], [], [], []
    for ep in range(args.episodes):
        seed = args.fixed_seed if args.fixed_seed is not None else args.seed_offset + ep
        ok, steps, visited = run_rl(model, env, seed, greedy=args.greedy)
        opt = optimal_steps(env.grid, (1, 1))
        rl_succ.append(ok); rl_ratio.append(steps / opt if ok else np.nan)
        ok2, steps2 = run_algo(cfg["h"], cfg["w"], cfg["key_door"], seed, env.max_steps)
        al_succ.append(ok2); al_ratio.append(steps2 / opt if ok2 else np.nan)
        if ep < args.render:
            print(f"\n--- episode {ep} seed {seed}: {'成功' if ok else '失败'} "
                  f"{steps} 步 (最优 {opt}) ---")
            print(render(env.env, visited))

    print(f"\n{'':12s}{'成功率':>8s}{'步数/最优':>12s}")
    print(f"{'PPO+LSTM':12s}{np.mean(rl_succ):8.2f}{np.nanmean(rl_ratio):12.2f}")
    print(f"{'算法 agent':12s}{np.mean(al_succ):8.2f}{np.nanmean(al_ratio):12.2f}")


if __name__ == "__main__":
    main()
