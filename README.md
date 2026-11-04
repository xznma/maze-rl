# maze-rl

完美迷宫 + 钥匙/门，以及三种走迷宫的 agent：算法、LLM、强化学习（PPO+LSTM）。

## 迷宫规则

- 递归回溯生成**完美迷宫**（任意两点之间有且仅有一条路径）
- 起点 `S`、终点 `E`；起点到终点的唯一路径中段有一扇门 `D`
- 钥匙 `K` 散落在不穿门就能到达的区域（保证有解）
- agent 需要找到钥匙、开门、到达终点

## 文件

| 文件 | 说明 |
|---|---|
| `maze_game.py` | 迷宫生成、`MazeEnv` 环境、算法 agent（BFS + 前沿探索），1000/1000 随机迷宫通关 |
| `llm_agent.py` | 用 Claude 当大脑的 agent，通过 `look` / `move` 两个工具和环境交互 |

## 运行

```bash
# 算法 agent
python maze_game.py --seed 7 --h 8 --w 16
python maze_game.py --animate          # 终端动画，带迷雾

# LLM agent（需要 pip install anthropic 和 ANTHROPIC_API_KEY）
python llm_agent.py --seed 7 --h 6 --w 10
```

## 路线图

- [x] 迷宫生成 + 算法 agent
- [x] LLM agent
- [ ] RL 环境包装（`env.py`）
- [ ] PPO + LSTM（`model.py` / `ppo.py` / `train.py` / `eval.py`）
- [ ] 课程学习：固定小迷宫 → 随机迷宫 → 加钥匙门
