"""
PPO + LSTM：rollout 采样、GAE、按整条序列做 minibatch 的更新。

和普通 PPO 的三点不同：
  1. minibatch 的单位是"环境序列"而不是单步，保持时间顺序
  2. 记录 rollout 开始时的隐状态 h0，更新时从 h0 重新前向整条序列
  3. 序列中 first=1 的位置重置隐状态（和采样时一致）
超时（truncated）的处理：把 γ·V(s_final) 加到该步奖励里，然后当作 done。
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class Rollout:
    def __init__(self, T, N, obs_dim):
        self.obs = torch.zeros(T, N, obs_dim)
        self.actions = torch.zeros(T, N, dtype=torch.long)
        self.logprobs = torch.zeros(T, N)
        self.values = torch.zeros(T, N)
        self.rewards = torch.zeros(T, N)
        self.dones = torch.zeros(T, N)      # 该步之后是否结束
        self.first = torch.zeros(T, N)      # 该步是否为一局第一步
        self.advantages = torch.zeros(T, N)
        self.returns = torch.zeros(T, N)


@torch.no_grad()
def collect(model, vec_env, obs, state, first, T, gamma):
    """采样 T 步。obs/state/first 是上一次 rollout 结束时的状态，返回更新后的。"""
    N = vec_env.n
    buf = Rollout(T, N, obs.shape[-1])
    h0 = (state[0].clone(), state[1].clone())
    episode_infos = []

    for t in range(T):
        obs_t = torch.as_tensor(obs)
        logits, value, state = model(obs_t[None], state, first[None])
        dist = Categorical(logits=logits[0])
        action = dist.sample()

        next_obs, reward, done, infos = vec_env.step(action.numpy())
        reward = torch.as_tensor(reward)
        for i, info in enumerate(infos):
            if done[i]:
                episode_infos.append(info)
                if info["truncated"]:        # 超时：用 V(final_obs) bootstrap
                    fo = torch.as_tensor(info["final_obs"])[None, None]
                    _, v_fin, _ = model(fo, (state[0][:, i:i + 1], state[1][:, i:i + 1]),
                                        torch.zeros(1, 1))
                    reward[i] += gamma * v_fin[0, 0]

        buf.obs[t] = obs_t
        buf.actions[t] = action
        buf.logprobs[t] = dist.log_prob(action)
        buf.values[t] = value[0]
        buf.rewards[t] = reward
        buf.dones[t] = torch.as_tensor(done)
        buf.first[t] = first

        obs = next_obs
        first = torch.as_tensor(done)

    # bootstrap 最后一步
    _, last_value, _ = model(torch.as_tensor(obs)[None], state, first[None])
    return buf, h0, obs, state, first, last_value[0], episode_infos


def compute_gae(buf, last_value, gamma, lam):
    T = buf.rewards.shape[0]
    gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else buf.values[t + 1]
        nonterminal = 1.0 - buf.dones[t]
        delta = buf.rewards[t] + gamma * next_v * nonterminal - buf.values[t]
        gae = delta + gamma * lam * nonterminal * gae
        buf.advantages[t] = gae
    buf.returns = buf.advantages + buf.values


def update(model, optimizer, buf, h0, epochs, n_minibatches, clip, vf_coef, ent_coef,
           max_grad_norm):
    T, N = buf.rewards.shape
    env_idx = np.arange(N)
    mb_size = max(1, N // n_minibatches)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    n_updates = 0

    for _ in range(epochs):
        np.random.shuffle(env_idx)
        for s in range(0, N, mb_size):
            idx = torch.as_tensor(env_idx[s:s + mb_size])
            state = (h0[0][:, idx], h0[1][:, idx])
            logits, values, _ = model(buf.obs[:, idx], state, buf.first[:, idx])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(buf.actions[:, idx])
            ratio = torch.exp(logp - buf.logprobs[:, idx])

            adv = buf.advantages[:, idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss = -torch.min(ratio * adv,
                                 torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
            v_loss = F.mse_loss(values, buf.returns[:, idx])
            entropy = dist.entropy().mean()
            loss = pg_loss + vf_coef * v_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += pg_loss.item()
            stats["value_loss"] += v_loss.item()
            stats["entropy"] += entropy.item()
            stats["approx_kl"] += (buf.logprobs[:, idx] - logp).mean().item()
            n_updates += 1
    return {k: v / n_updates for k, v in stats.items()}
