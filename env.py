"""
RL 环境包装：把 MazeEnv 包成 reset()/step() 接口，并定义观测编码和奖励。

两种观测模式:

local (阶段 0–4, 25 维):
    [0:20]  四周 4 格 (上/下/左/右) × 5 类 one-hot (墙/空/终点/门/钥匙)
    [20]    是否持有钥匙
    [21:25] 上一步动作 one-hot
    不含绝对坐标 —— agent 必须靠 LSTM 记忆来定位自己。

map (阶段 5+):
    以 agent 为中心的 (2R+1)×(2R+1) 窗口，6 个通道: 墙 / 已见地面 / 走过 / 终点 / 钥匙 / 门，
    只画 agent 自己看见过的格子（环境替 agent 维护一张"已探索地图"），
    再拼上 [是否持钥匙, 上一步动作 one-hot]。
    记忆负担从 LSTM 里拿出来交给这张图，LSTM 只需要记短期的东西（比如正在往哪走）。

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
OBS_DIM = 4 * N_CELL_TYPES + 1 + N_ACTIONS          # local 模式
MAP_CH = 6                                           # map 模式通道数
MAP_CHANNEL = {WALL: 0, " ": 1, "S": 1, EXIT: 3, KEY: 4, DOOR: 5}   # 2 = 走过
VEC_DIM = 1 + N_ACTIONS


def map_obs_dim(radius):
    return MAP_CH * (2 * radius + 1) ** 2 + VEC_DIM


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
                 rng_seed=0, obs_mode="local", map_radius=12, size_range=None):
        self.h, self.w = h, w
        self.size_range = size_range              # (lo, hi): 每局随机抽 h、w
        self.with_key_door = with_key_door
        self.fixed_seed = fixed_seed              # None = 每局随机换迷宫
        self.rng = random.Random(rng_seed)
        self._max_steps_arg = max_steps
        self.obs_mode = obs_mode
        self.R = map_radius
        self.obs_dim = OBS_DIM if obs_mode == "local" else map_obs_dim(map_radius)
        self.env = None

    # ---- 接口 ----
    def reset(self, seed=None):
        if self.size_range:
            self.h = self.rng.randint(*self.size_range)
            self.w = self.rng.randint(*self.size_range)
        s = seed if seed is not None else (
            self.fixed_seed if self.fixed_seed is not None else self.rng.getrandbits(31))
        grid, start = build_level(self.h, self.w, s, self.with_key_door)
        self.env = MazeEnv(grid, start)
        self.max_steps = self._max_steps_arg or 4 * (2 * self.h * self.w - 1)
        self.visited = {start}
        self.last_action = -1
        self.maze_seed = s
        self.t = 0
        if self.obs_mode == "map":
            H, W, R = len(grid), len(grid[0]), self.R
            self.kmap = np.zeros((MAP_CH, H + 2 * R, W + 2 * R), dtype=np.float32)
            self._see()
            self.kmap[2, start[0] + R, start[1] + R] = 1.0
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
        if self.obs_mode == "map":
            self._see()
            self.kmap[2, env.pos[0] + self.R, env.pos[1] + self.R] = 1.0
        # 注意 env.steps 只在真的移动时 +1，这里要按动作次数计超时
        self.t += 1
        truncated = (not env.done) and self.t >= self.max_steps
        done = env.done or truncated
        info = {"success": env.done, "truncated": truncated, "steps": self.t,
                "size": (self.h, self.w)}
        if done:
            self.t = 0
        return self._obs(), reward, done, info

    # ---- 观测编码 ----
    def _see(self):
        """把当前位置和四周四格写进已探索地图（重写，因为钥匙/门用过后会变成地面）。"""
        r, c = self.env.pos
        R = self.R
        for dr, dc in [(0, 0)] + DIRS:
            rr, cc = r + dr, c + dc
            ch = self.env.grid[rr][cc]
            self.kmap[[0, 1, 3, 4, 5], rr + R, cc + R] = 0.0
            self.kmap[MAP_CHANNEL[ch], rr + R, cc + R] = 1.0

    def _vec(self):
        v = np.zeros(VEC_DIM, dtype=np.float32)
        v[0] = float(self.env.has_key)
        if self.last_action >= 0:
            v[1 + self.last_action] = 1.0
        return v

    def _obs(self):
        if self.obs_mode == "map":
            r, c = self.env.pos
            S = 2 * self.R + 1
            window = self.kmap[:, r:r + S, c:c + S]          # kmap 已经 pad 了 R
            return np.concatenate([window.ravel(), self._vec()])
        env = self.env
        r, c = env.pos
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for i, (dr, dc) in enumerate(DIRS):
            obs[i * N_CELL_TYPES + CELL_TYPES[env.grid[r + dr][c + dc]]] = 1.0
        obs[20:] = self._vec()
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
        self.obs_dim = self.envs[0].obs_dim

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
