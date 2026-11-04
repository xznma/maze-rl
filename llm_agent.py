"""
用大语言模型（Claude）当迷宫 agent 的"大脑"。

架构：
  LLM 不直接看地图，只能通过两个工具和环境交互：
    look()      -> 返回当前位置、四周格子、是否持有钥匙
    move(dir)   -> 朝一个方向走一步，返回结果
  记忆（走过哪些格子、看到过什么）由 LLM 自己在对话上下文里维护，
  决策（去哪、怎么回头）也由 LLM 自己推理。

运行:
  pip install anthropic
  export ANTHROPIC_API_KEY=...
  python llm_agent.py --seed 7 --h 6 --w 10
"""
import argparse

import anthropic
from anthropic import beta_tool

from maze_game import MazeEnv, build_level, render

DIR_MAP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
NAMES = {"#": "wall", " ": "open", "S": "start", "E": "exit", "D": "door", "K": "key"}

SYSTEM = """\
You are navigating a maze you cannot see. You start at S and must reach E.
Rules:
- Somewhere on the only path to E there is a locked door (D). You cannot pass it without the key.
- A key (K) is somewhere in the part of the maze reachable without passing the door. Walk onto it to pick it up.
- The maze is a perfect maze: no loops, exactly one route between any two cells. Dead ends are common.
Use `look` to see the four neighbouring cells and `move` to step. Keep a mental map of visited cells
and unexplored openings; prefer the nearest unexplored opening, and backtrack when you hit a dead end.
Once you have the key, head for the door, then the exit. Stop calling tools once you reach E.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=6)
    ap.add_argument("--w", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    grid, start = build_level(args.h, args.w, args.seed)
    env = MazeEnv(grid, start)
    visited = {start}

    # ---- 工具：LLM 能做的全部事情 ----
    @beta_tool
    def look() -> str:
        """Observe your current cell and the four neighbouring cells (up/down/left/right)."""
        r, c = env.pos
        parts = [f"position=({r},{c})", f"has_key={env.has_key}"]
        for name, (dr, dc) in DIR_MAP.items():
            parts.append(f"{name}={NAMES[env.grid[r + dr][c + dc]]}")
        return ", ".join(parts)

    @beta_tool
    def move(direction: str) -> str:
        """Move one cell in a direction.

        Args:
            direction: One of "up", "down", "left", "right".
        """
        if direction not in DIR_MAP:
            return f"invalid direction {direction!r}"
        if env.steps >= args.max_steps:
            return "step limit reached; stop."
        msg = env.step(DIR_MAP[direction])
        visited.add(env.pos)
        r, c = env.pos
        return f"{msg or 'ok'}; now at ({r},{c}), has_key={env.has_key}, done={env.done}"

    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model="claude-opus-5-5",
        max_tokens=16000,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        tools=[look, move],
        messages=[{"role": "user", "content": "Find the key, open the door, reach the exit."}],
        max_iterations=args.max_steps + 50,
    )
    for message in runner:
        for block in message.content:
            if block.type == "text" and block.text.strip():
                print("LLM:", block.text.strip())
        if env.done:
            break

    print(render(env, visited))
    print(f"\n结果: {'成功通关' if env.done else '失败'}，共 {env.steps} 步")


if __name__ == "__main__":
    main()
