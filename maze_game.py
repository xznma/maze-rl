"""
完美迷宫 + 钥匙门 + 探索型 Agent

运行:  python maze_game.py              # 默认 15x25，打印结果
       python maze_game.py --animate    # 终端动画看 agent 走
       python maze_game.py --seed 42 --h 10 --w 20
"""
import argparse
import random
import time
from collections import deque

WALL, PATH, START, EXIT, DOOR, KEY = "#", " ", "S", "E", "D", "K"
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ---------------------------------------------------------------------------
# 1. 生成完美迷宫（递归回溯 / DFS）
#    完美迷宫 = 任意两点之间有且仅有一条路径（生成树），
#    所以 起点->终点 的路径是唯一的，在上面放门就一定是“必经之路”。
# ---------------------------------------------------------------------------
def generate_maze(h, w, rng):
    H, W = 2 * h + 1, 2 * w + 1
    grid = [[WALL] * W for _ in range(H)]
    stack = [(1, 1)]
    grid[1][1] = PATH
    while stack:
        r, c = stack[-1]
        nbrs = [(r + 2 * dr, c + 2 * dc, dr, dc) for dr, dc in DIRS
                if 0 < r + 2 * dr < H and 0 < c + 2 * dc < W
                and grid[r + 2 * dr][c + 2 * dc] == WALL]
        if nbrs:
            nr, nc, dr, dc = rng.choice(nbrs)
            grid[r + dr][c + dc] = PATH      # 打通中间的墙
            grid[nr][nc] = PATH
            stack.append((nr, nc))
        else:
            stack.pop()
    return grid


def bfs(grid, start, blocked=frozenset()):
    """返回 parent 字典（可达集合 + 回溯路径用）。"""
    parent = {start: None}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            n = (r + dr, c + dc)
            if n not in parent and n not in blocked and grid[n[0]][n[1]] != WALL:
                parent[n] = (r, c)
                q.append(n)
    return parent


def trace(parent, goal):
    path = []
    while goal is not None:
        path.append(goal)
        goal = parent[goal]
    return path[::-1]


# ---------------------------------------------------------------------------
# 2. 放置 起点 / 终点 / 门 / 钥匙
# ---------------------------------------------------------------------------
def build_level(h, w, seed=None, with_key_door=True):
    rng = random.Random(seed)
    grid = generate_maze(h, w, rng)
    start, exit_ = (1, 1), (2 * h - 1, 2 * w - 1)

    if not with_key_door:                    # 简化版：只有起点终点
        grid[start[0]][start[1]] = START
        grid[exit_[0]][exit_[1]] = EXIT
        return grid, start

    # 唯一路径
    path = trace(bfs(grid, start), exit_)

    # 门：放在路径中段的某个格子上（不能是起点/终点）
    # 钥匙：必须在“不穿门就能到达”的区域（否则无解），
    # 且不在主路径上，优先放在离主路径较远的死胡同，增加探索难度
    on_path = set(path)
    inner = list(range(1, len(path) - 1))
    mid = [i for i in inner if len(path) // 3 <= i <= 2 * len(path) // 3]
    rng.shuffle(mid)
    others = [i for i in reversed(inner) if i not in mid]
    for i in mid + others:                   # 优先中段，不行就往后挪
        door = path[i]
        reachable = bfs(grid, start, blocked={door})
        candidates = [p for p in reachable if p not in on_path and p != start] \
            or [p for p in reachable if p != start]
        if candidates:
            break
    else:
        raise ValueError("迷宫太小，放不下钥匙和门，请调大尺寸")
    dist_from_path = bfs_multi(grid, [p for p in path if p in reachable], blocked={door})
    dead_ends = [p for p in candidates if sum(
        grid[p[0] + dr][p[1] + dc] != WALL for dr, dc in DIRS) == 1]
    pool = dead_ends or candidates
    pool.sort(key=lambda p: dist_from_path.get(p, 0), reverse=True)
    key = rng.choice(pool[:max(1, len(pool) // 3)])   # 在最远的 1/3 里随机

    grid[start[0]][start[1]] = START
    grid[exit_[0]][exit_[1]] = EXIT
    grid[door[0]][door[1]] = DOOR
    grid[key[0]][key[1]] = KEY
    return grid, start


def bfs_multi(grid, sources, blocked):
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            n = (r + dr, c + dc)
            if n not in dist and n not in blocked and grid[n[0]][n[1]] != WALL:
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist


# ---------------------------------------------------------------------------
# 3. 游戏环境：agent 只能看到自己周围一格（部分可观测）
# ---------------------------------------------------------------------------
class MazeEnv:
    def __init__(self, grid, start):
        self.grid = [row[:] for row in grid]
        self.pos = start
        self.has_key = False
        self.done = False
        self.steps = 0

    def observe(self):
        """返回当前位置及上下左右四格的内容。"""
        r, c = self.pos
        view = {self.pos: self.grid[r][c]}
        for dr, dc in DIRS:
            view[(r + dr, c + dc)] = self.grid[r + dr][c + dc]
        return view

    def step(self, move):
        r, c = self.pos[0] + move[0], self.pos[1] + move[1]
        cell = self.grid[r][c]
        if cell == WALL:
            return "撞墙"
        if cell == DOOR and not self.has_key:
            return "门是锁着的"
        self.pos = (r, c)
        self.steps += 1
        if cell == KEY:
            self.has_key = True
            self.grid[r][c] = PATH
            return "捡到钥匙!"
        if cell == DOOR:
            self.grid[r][c] = PATH
            return "用钥匙打开了门!"
        if cell == EXIT:
            self.done = True
            return "到达终点!"
        return ""


# ---------------------------------------------------------------------------
# 4. Agent：不知道地图，靠记忆 + 前沿探索（frontier exploration）
#    决策规则（每一步重新规划）：
#      a) 已知终点且能走到 -> 去终点
#      b) 否则 -> 去最近的“前沿格”（已知可走、但旁边还有未知格）
#    门在没钥匙时视为不可通行；拿到钥匙后自动变为可通行，
#    因此“先找钥匙、再开门、再出去”是自然涌现的行为，不需要硬编码。
# ---------------------------------------------------------------------------
class ExplorerAgent:
    def __init__(self):
        self.memory = {}          # (r,c) -> 看到的格子内容
        self.has_key = False

    def passable(self, p):
        cell = self.memory.get(p)
        if cell is None or cell == WALL:
            return False
        if cell == DOOR and not self.has_key:
            return False
        return True

    def is_frontier(self, p):
        return any((p[0] + dr, p[1] + dc) not in self.memory for dr, dc in DIRS)

    def act(self, pos, view, has_key):
        self.memory.update(view)
        self.has_key = has_key

        # 在已知地图上 BFS，找最近的目标
        parent = {pos: None}
        q = deque([pos])
        exit_target = key_target = frontier_target = None
        while q:
            cur = q.popleft()
            if self.memory.get(cur) == EXIT:
                exit_target = cur
                break
            if key_target is None and self.memory.get(cur) == KEY:
                key_target = cur          # 看到过钥匙（哪怕没站上去）就去捡
            if frontier_target is None and cur != pos and self.is_frontier(cur):
                frontier_target = cur
            for dr, dc in DIRS:
                n = (cur[0] + dr, cur[1] + dc)
                if n not in parent and self.passable(n):
                    parent[n] = cur
                    q.append(n)

        target = exit_target or key_target or frontier_target
        if target is None:
            return None           # 无路可走（理论上不会发生）
        nxt = trace(parent, target)[1]
        return (nxt[0] - pos[0], nxt[1] - pos[1])


# ---------------------------------------------------------------------------
# 5. 运行 & 可视化
# ---------------------------------------------------------------------------
def render(env, visited, memory=None):
    out = []
    for r, row in enumerate(env.grid):
        line = ""
        for c, ch in enumerate(row):
            p = (r, c)
            if p == env.pos:
                line += "@"
            elif memory is not None and p not in memory:
                line += "░"           # 迷雾：agent 还没看到
            elif ch in (START, EXIT, DOOR, KEY):
                line += ch
            elif p in visited:
                line += "·"
            else:
                line += "█" if ch == WALL else " "
        out.append(line)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=15)
    ap.add_argument("--w", type=int, default=25)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--delay", type=float, default=0.03)
    args = ap.parse_args()

    grid, start = build_level(args.h, args.w, args.seed)
    env, agent = MazeEnv(grid, start), ExplorerAgent()
    print("初始迷宫：\n" + render(env, set()) + "\n")

    visited = {start}
    max_steps = 20 * args.h * args.w
    while not env.done and env.steps < max_steps:
        move = agent.act(env.pos, env.observe(), env.has_key)
        if move is None:
            print("Agent 卡住了")
            break
        msg = env.step(move)
        visited.add(env.pos)
        if msg and msg not in ("撞墙", "门是锁着的"):
            print(f"[第 {env.steps} 步] {msg}")
        if args.animate:
            print("\033[H\033[J" + render(env, visited, agent.memory))
            print(f"步数 {env.steps}  钥匙 {'有' if env.has_key else '无'}")
            time.sleep(args.delay)

    print("\n最终轨迹（· = 走过的格子）：\n" + render(env, visited))
    print(f"\n结果: {'成功通关' if env.done else '失败'}，共 {env.steps} 步")


if __name__ == "__main__":
    main()
