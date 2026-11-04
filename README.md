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
| `env.py` | RL 环境包装：25 维局部观测（四周 4 格 + 是否持钥匙 + 上一步动作，无绝对坐标）、奖励设计、向量化环境 |
| `model.py` | Actor-Critic：MLP → LSTM(128) → 策略头 + 价值头，按 episode 边界重置隐状态 |
| `ppo.py` | PPO + LSTM：rollout 采样、GAE、按整条序列做 minibatch、超时 bootstrap |
| `train.py` | 训练主循环 + 四个课程阶段 |
| `eval.py` | 在没见过的迷宫上评估，和算法 agent 并排对比 |

## 运行

```bash
# 算法 agent
python maze_game.py --seed 7 --h 8 --w 16
python maze_game.py --animate          # 终端动画，带迷雾

# LLM agent（需要 pip install anthropic 和 ANTHROPIC_API_KEY）
python llm_agent.py --seed 7 --h 6 --w 10

# PPO + LSTM（需要 pip install torch numpy）
python train.py --stage 0                       # 固定 3x3，无钥匙门：验证代码
python train.py --stage 1                       # 固定 5x5，无钥匙门
python train.py --stage 2                       # 随机 5x5，无钥匙门
python train.py --stage 3 --init ckpt/stage2.pt # 随机 5x5，有钥匙门，接着练
python eval.py ckpt/stage2.pt --episodes 100 --render 3
```

## 课程进度

| 阶段 | 设定 | 结果 |
|---|---|---|
| 0 | 固定 3×3，无钥匙门 | ✅ 8 万步达到 100% 成功率；贪心评估 21 步 / 最优 16 步 |
| 1 | 固定 5×5，无钥匙门 | 训练中 |
| 2 | 随机 5×5，无钥匙门 | — |
| 3 | 随机 5×5，有钥匙门 | — |

## PPO+LSTM 设计要点

- **观测不含绝对坐标**，agent 必须靠 LSTM 记忆来定位，否则 LSTM 没意义
- **奖励**：终点 +1，钥匙 +0.3，新格子 +0.02（探索奖励，解决稀疏问题的关键），每步 −0.01，撞墙 −0.01
- **minibatch 以整条环境序列为单位**，保持时间顺序；更新时从 rollout 开始的 h₀ 重新前向
- **episode 边界重置隐状态**（`first` 掩码），采样和更新时都做，防止上一局记忆泄漏
- **超时**（truncated）用 γ·V(s_final) bootstrap，和真正到达终点区分开
