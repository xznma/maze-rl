"""
RL 环境包装：把 MazeEnv 包成 reset()/step() 接口，并定义观测编码和奖励。

观测 (25 维):
    [0:20]  四周 4 格 (上/下/左/右) × 5 类 one-hot (墙/空/终点/门/钥匙)
    [20]    是否持有钥匙
    [21:25] 上一步动作 one-hot
    不含绝对坐标 —— agent 必须靠 LSTM 记忆来定位自己。

奖励:
    到达终点  +1.0
    捡到钥匙  +0.3
    新格子    +0.02   (每局重置)
    每步      -0.01
    撞墙/锁门 -0.01
"""
import random

import numpy as np

from maze_game import DIRS, DOOR, EXIT, KEY, WALL, MazeEnv, bfs, build_level, trace

CELL_TYPES = {WALL: 0, " ": 1, "S": 1, EXIT: 2, DOOR: 3, KEY: 4}
N_CELL_TYPES = 5
N_ACTIONS = 4
OBS_DIM = 4 * N_CELL_TYPES + 1 + N_ACTIONS

R_EXIT, R_KEY, R_NEW_CELL, R_STEP, R_BUMP = 1.0, 0.3, 0.02, -0.01, -0.01


def optimal_steps(grid, start):
    """最短路长度（有钥匙门时 = S→K→E）。用于评估时算"绕路比"。"""
    cells = {grid[r][c]: (r, c) for r in range(len(grid)) for c in range(len(grid[0]))
             if grid[r][c] in (EXIT, KEY)}
    exit_ = cells[EXIT]
    if KEY not in cells:
        return len(trace(bfs(grid, start), exit_)) - 1
    key = cells[KEY]
    return (len(trace(bfs(grid, start), key)) - 1
            + len(trace(bfs(grid, key), exit_)) - 1)


class MazeRLEnv:
    def __init__(self, h, w, with_key_door=False, fixed_seed=None, max_steps=None,
                 rng_seed=0):
        self.h, self.w = h, w
        self.with_key_door = with_key_door
        self.fixed_seed = fixed_seed              # None = 每局随机换迷宫
        self.rng = random.Random(rng_seed)
        self.max_steps = max_steps or 4 * (2 * h * w - 1)   # 4 × 可走格子数
        self.env = None

    # ---- 接口 ----
    def reset(self, seed=None):
        s = seed if seed is not None else (
            self.fixed_seed if self.fixed_seed is not None else self.rng.getrandbits(31))
        grid, start = build_level(self.h, self.w, s, self.with_key_door)
        self.env = MazeEnv(grid, start)
        self.visited = {start}
        self.last_action = -1
        self.maze_seed = s
        self.t = 0
        return self._obs()

    def step(self, action):
        env = self.env
        msg = env.step(DIRS[action])
        self.last_action = action
        reward = R_STEP
        if msg in ("撞墙", "门是锁着的"):
            reward += R_BUMP
        else:
            if env.pos not in self.visited:
                self.visited.add(env.pos)
                reward += R_NEW_CELL
            if msg == "捡到钥匙!":
                reward += R_KEY
            elif msg == "到达终点!":
                reward += R_EXIT
        # 注意 env.steps 只在真的移动时 +1，这里要按动作次数计超时
        self.t += 1
        truncated = (not env.done) and self.t >= self.max_steps
        done = env.done or truncated
        info = {"success": env.done, "truncated": truncated, "steps": self.t}
        if done:
            self.t = 0
        return self._obs(), reward, done, info

    # ---- 观测编码 ----
    def _obs(self):
        env = self.env
        r, c = env.pos
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for i, (dr, dc) in enumerate(DIRS):
            obs[i * N_CELL_TYPES + CELL_TYPES[env.grid[r + dr][c + dc]]] = 1.0
        obs[20] = float(env.has_key)
        if self.last_action >= 0:
            obs[21 + self.last_action] = 1.0
        return obs

    # 给 eval 用
    @property
    def grid(self):
        return self.env.grid

    @property
    def pos(self):
        return self.env.pos


class VecEnv:
    """最简单的同步向量化环境：N 个 MazeRLEnv，done 时自动 reset。"""

    def __init__(self, n, **env_kwargs):
        self.envs = [MazeRLEnv(rng_seed=i, **env_kwargs) for i in range(n)]
        self.n = n

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        obs, rews, dones, infos = [], [], [], []
        for env, a in zip(self.envs, actions):
            o, r, d, info = env.step(int(a))
            if d:
                info["final_obs"] = o          # 超时 bootstrap 要用
                o = env.reset()
            obs.append(o); rews.append(r); dones.append(d); infos.append(info)
        return (np.stack(obs), np.array(rews, dtype=np.float32),
                np.array(dones, dtype=np.float32), infos)
