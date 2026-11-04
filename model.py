"""
Actor-Critic 网络：obs → 编码器 → LSTM → (策略 logits, 价值 V)

编码器按观测模式选:
    local: MLP(25 → 64)
    map:   CNN 处理 6×(2R+1)×(2R+1) 的已探索地图 → 拼上 [持钥匙, 上一步动作] → Linear(128)

forward 按时间步循环，`first[t, n] == 1` 表示 obs[t, n] 是一局的第一步，
此时该环境的 LSTM 隐状态在处理前清零 —— 防止上一局的记忆泄漏到下一局。
"""
import torch
import torch.nn as nn

from env import MAP_CH, N_ACTIONS, OBS_DIM, VEC_DIM, map_obs_dim


class ActorCritic(nn.Module):
    def __init__(self, obs_mode="local", map_radius=12, n_actions=N_ACTIONS, hidden=64,
                 lstm=128):
        super().__init__()
        self.obs_mode, self.R, self.lstm_size = obs_mode, map_radius, lstm
        if obs_mode == "local":
            self.obs_dim = OBS_DIM
            self.encoder = nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.ReLU())
            enc_out = hidden
        else:
            self.obs_dim = map_obs_dim(map_radius)
            S = 2 * map_radius + 1
            self.cnn = nn.Sequential(
                nn.Conv2d(MAP_CH, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten())
            with torch.no_grad():
                cnn_out = self.cnn(torch.zeros(1, MAP_CH, S, S)).shape[1]
            enc_out = 128
            self.fc = nn.Sequential(nn.Linear(cnn_out + VEC_DIM, enc_out), nn.ReLU())
        self.lstm = nn.LSTM(enc_out, lstm)
        self.policy = nn.Linear(lstm, n_actions)
        self.value = nn.Linear(lstm, 1)
        for name, p in self.lstm.named_parameters():
            (nn.init.orthogonal_ if "weight" in name else nn.init.zeros_)(p)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def initial_state(self, n):
        z = torch.zeros(1, n, self.lstm_size)
        return (z, z.clone())

    def encode(self, obs):
        """obs [T, N, obs_dim] → [T, N, enc_out]"""
        if self.obs_mode == "local":
            return self.encoder(obs)
        T, N, _ = obs.shape
        S = 2 * self.R + 1
        flat = obs.reshape(T * N, -1)
        grid, vec = flat[:, :-VEC_DIM].reshape(T * N, MAP_CH, S, S), flat[:, -VEC_DIM:]
        return self.fc(torch.cat([self.cnn(grid), vec], dim=1)).reshape(T, N, -1)

    def forward(self, obs, state, first):
        """
        obs:   [T, N, obs_dim]
        state: (h, c), 各 [1, N, lstm]
        first: [T, N]  1 = 该步是新一局的第一步
        返回 logits [T, N, A], values [T, N], 新 state
        """
        x = self.encode(obs)
        h, c = state
        outs = []
        for t in range(x.shape[0]):
            mask = (1.0 - first[t]).view(1, -1, 1)
            h, c = h * mask, c * mask
            out, (h, c) = self.lstm(x[t:t + 1], (h, c))
            outs.append(out)
        feat = torch.cat(outs, dim=0)
        return self.policy(feat), self.value(feat).squeeze(-1), (h, c)


def model_from_checkpoint(ck):
    """从 train.py 存的 checkpoint 重建网络（兼容早期没有 obs 字段的 checkpoint）。"""
    cfg = ck["args"]
    model = ActorCritic(cfg.get("obs", "local"), cfg.get("map_radius", 12))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg
