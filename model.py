"""
Actor-Critic 网络：obs → MLP → LSTM → (策略 logits, 价值 V)

forward 按时间步循环，`first[t, n] == 1` 表示 obs[t, n] 是一局的第一步，
此时该环境的 LSTM 隐状态在处理前清零 —— 防止上一局的记忆泄漏到下一局。
"""
import torch
import torch.nn as nn

from env import N_ACTIONS, OBS_DIM


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=64, lstm=128):
        super().__init__()
        self.lstm_size = lstm
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, lstm)
        self.policy = nn.Linear(lstm, n_actions)
        self.value = nn.Linear(lstm, 1)
        for name, p in self.lstm.named_parameters():
            (nn.init.orthogonal_ if "weight" in name else nn.init.zeros_)(p)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def initial_state(self, n):
        z = torch.zeros(1, n, self.lstm_size)
        return (z, z.clone())

    def forward(self, obs, state, first):
        """
        obs:   [T, N, obs_dim]
        state: (h, c), 各 [1, N, lstm]
        first: [T, N]  1 = 该步是新一局的第一步
        返回 logits [T, N, A], values [T, N], 新 state
        """
        x = self.encoder(obs)
        h, c = state
        outs = []
        for t in range(x.shape[0]):
            mask = (1.0 - first[t]).view(1, -1, 1)
            h, c = h * mask, c * mask
            out, (h, c) = self.lstm(x[t:t + 1], (h, c))
            outs.append(out)
        feat = torch.cat(outs, dim=0)
        return self.policy(feat), self.value(feat).squeeze(-1), (h, c)
