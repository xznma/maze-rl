"""
生成一个自包含的网页回放：同一张迷宫上，PPO+LSTM agent 和算法 agent 并排走。

  python viz.py ckpt/stage3.pt --seed 1000003 --out replay.html
  python viz.py ckpt/stage4.pt --seed 1000010 --h 8 --w 8 --out replay_8x8.html

打开 replay.html 即可，不需要服务器。
"""
import argparse
import json

import torch

from env import MazeRLEnv, optimal_steps
from maze_game import DIRS, ExplorerAgent, MazeEnv, build_level
from model import model_from_checkpoint


def record_rl(model, h, w, key_door, seed):
    env = MazeRLEnv(h, w, key_door, obs_mode=model.obs_mode, map_radius=model.R)
    obs = env.reset(seed)
    state, first = model.initial_state(1), torch.ones(1)
    steps, done = [], False
    with torch.no_grad():
        while not done:
            logits, _, state = model(torch.as_tensor(obs)[None, None], state, first[None])
            a = int(logits[0, 0].argmax())
            prev = env.pos
            obs, _, done, info = env.step(a)
            steps.append(_rec(env.env, prev))
            first = torch.zeros(1)
    return steps, info["success"]


def record_algo(grid, start, max_steps):
    env, agent = MazeEnv([r[:] for r in grid], start), ExplorerAgent()
    steps = []
    while not env.done and len(steps) < max_steps:
        prev = env.pos
        env.step(agent.act(env.pos, env.observe(), env.has_key))
        steps.append(_rec(env, prev))
    return steps, env.done


def _rec(env, prev):
    r, c = env.pos
    return {"r": r, "c": c, "k": int(env.has_key), "e": "bump" if env.pos == prev else ""}


def annotate_events(steps, grid):
    """根据初始地图标注捡钥匙/开门/到终点事件。"""
    got_key = False
    for s in steps:
        cell = grid[s["r"]][s["c"]]
        if cell == "K" and not got_key:
            s["e"], got_key = "key", True
        elif cell == "D" and s["e"] != "bump" and got_key:
            s["e"] = "door"
            got_key = None          # 门只标一次
        elif cell == "E":
            s["e"] = "exit"
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seed", type=int, default=1_000_003)
    ap.add_argument("--h", type=int); ap.add_argument("--w", type=int)
    ap.add_argument("--out", default="replay.html")
    args = ap.parse_args()

    model, cfg = model_from_checkpoint(torch.load(args.ckpt))
    h, w, key_door = args.h or cfg["h"], args.w or cfg["w"], cfg["key_door"]

    grid, start = build_level(h, w, args.seed, key_door)
    max_steps = 4 * (2 * h * w - 1)
    rl_steps, rl_ok = record_rl(model, h, w, key_door, args.seed)
    al_steps, al_ok = record_algo(grid, start, max_steps)

    data = {
        "seed": args.seed, "h": h, "w": w, "key_door": key_door,
        "grid": ["".join(r) for r in grid], "start": list(start),
        "optimal": optimal_steps(grid, start), "max_steps": max_steps,
        "agents": [
            {"name": "PPO + LSTM" + ("（地图观测）" if model.obs_mode == "map" else ""), "steps": annotate_events(rl_steps, grid), "success": rl_ok},
            {"name": "算法 agent", "steps": annotate_events(al_steps, grid), "success": al_ok},
        ],
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}: PPO {len(rl_steps)} 步 {'成功' if rl_ok else '失败'}, "
          f"算法 {len(al_steps)} 步 {'成功' if al_ok else '失败'}, 最优 {data['optimal']}")


TEMPLATE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Maze Replay</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>
/* layout: two boards side by side on one shared timeline; controls under them */
:root {
  --bg: #f3f1ec; --panel: #ffffff; --fg: #1d1c1a; --muted: #6f6b63; --line: #d9d5cc;
  --wall: #2b2a27; --floor: #faf9f6; --fog: rgba(60,58,52,.78);
  --trail: #cfd9e8; --agent: #1f5fbf; --key: #d9a21b; --door: #a0522d; --exit: #2e9e5b; --start: #8d8a82;
  --accent: #1f5fbf; --ok: #2e9e5b; --bad: #c63f3f;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
  --sans: "IBM Plex Sans", system-ui, -apple-system, sans-serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #16171a; --panel: #1f2126; --fg: #e8e6e1; --muted: #9a978f; --line: #33363d;
  --wall: #0c0d0f; --floor: #2a2d33; --fog: rgba(10,11,13,.82);
  --trail: #2f3f5a; --agent: #6ea8ff; --key: #f0c040; --door: #d2794f; --exit: #4fc97e; --start: #777;
  --accent: #6ea8ff; --ok: #4fc97e; --bad: #ff6b6b; color-scheme: dark;
}}
:root[data-theme="dark"] {
  --bg: #16171a; --panel: #1f2126; --fg: #e8e6e1; --muted: #9a978f; --line: #33363d;
  --wall: #0c0d0f; --floor: #2a2d33; --fog: rgba(10,11,13,.82);
  --trail: #2f3f5a; --agent: #6ea8ff; --key: #f0c040; --door: #d2794f; --exit: #4fc97e; --start: #777;
  --accent: #6ea8ff; --ok: #4fc97e; --bad: #ff6b6b; color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 var(--sans);
  padding-block: 24px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 1100px; margin: 0 auto; display: grid; gap: 20px; }
h1 { font: 600 22px/1.2 var(--sans); margin: 0; text-wrap: balance; }
.sub { color: var(--muted); font-family: var(--mono); font-size: 13px; }
.boards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
.board { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; display: grid; gap: 10px; min-width: 0; }
.board header { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.board h2 { font: 600 16px var(--sans); margin: 0; }
.status { font-family: var(--mono); font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; }
.status b { color: var(--fg); font-weight: 500; }
.status .ok { color: var(--ok); } .status .bad { color: var(--bad); }
canvas { width: 100%; max-width: 100%; height: auto; display: block; border-radius: 4px; image-rendering: pixelated; }
.events { display: flex; gap: 6px; flex-wrap: wrap; min-height: 24px; }
.ev { font: 12px var(--mono); padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
.ev.key { border-color: var(--key); color: var(--key); } .ev.door { border-color: var(--door); color: var(--door); }
.ev.exit { border-color: var(--exit); color: var(--exit); }
.controls { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; display: grid; gap: 12px; }
.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
button { font: 500 14px var(--sans); padding: 7px 14px; border-radius: 6px; border: 1px solid var(--line); background: var(--bg); color: var(--fg); cursor: pointer; }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
input[type=range] { flex: 1 1 160px; min-width: 120px; accent-color: var(--accent); }
label { font-size: 13px; color: var(--muted); display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.t { font-family: var(--mono); font-variant-numeric: tabular-nums; min-width: 9ch; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 13px; color: var(--muted); }
.legend span::before { content: ""; display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; background: var(--c); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head>
<body>
<main>
  <div>
    <h1 id="title">迷宫回放</h1>
    <div class="sub" id="sub"></div>
  </div>
  <div class="boards" id="boards"></div>
  <div class="controls">
    <div class="row">
      <button class="primary" id="play">播放</button>
      <button id="stepb">单步</button>
      <button id="reset">重置</button>
      <span class="t" id="tlabel">0 步</span>
      <input type="range" id="scrub" min="0" value="0" aria-label="进度">
    </div>
    <div class="row">
      <label>速度 <input type="range" id="speed" min="1" max="30" value="8" aria-label="速度"></label>
      <label><input type="checkbox" id="fog" checked> 迷雾（只显示 agent 看过的格子）</label>
      <label><input type="checkbox" id="trail" checked> 轨迹</label>
    </div>
    <div class="legend">
      <span style="--c: var(--agent)">agent</span><span style="--c: var(--trail)">走过</span>
      <span style="--c: var(--key)">钥匙 K</span><span style="--c: var(--door)">门 D</span>
      <span style="--c: var(--exit)">终点 E</span><span style="--c: var(--start)">起点 S</span>
      <span style="--c: var(--fog)">未知</span>
    </div>
  </div>
</main>
<script>
const D = __DATA__;
const H = D.grid.length, W = D.grid[0].length, CELL = 24;
const maxLen = Math.max(...D.agents.map(a => a.steps.length));
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

document.getElementById('sub').textContent =
  `${D.h}×${D.w} 房间格 · seed ${D.seed} · ${D.key_door ? '有钥匙门' : '无钥匙门'} · 最优 ${D.optimal} 步 · 上限 ${D.max_steps} 步`;

// 预计算每个 agent 在第 t 步时：位置、已看见格子、走过格子、是否持钥匙、事件
const states = D.agents.map(a => {
  const seen = new Set(), visited = new Set(), frames = [];
  const [sr, sc] = D.start;
  const look = (r, c) => { for (const [dr, dc] of [[0,0],[-1,0],[1,0],[0,-1],[0,1]]) seen.add((r+dr)*W + c+dc); };
  look(sr, sc); visited.add(sr*W + sc);
  frames.push({ r: sr, c: sc, k: 0, seen: new Set(seen), visited: new Set(visited), events: [] });
  const events = [];
  a.steps.forEach((s, i) => {
    look(s.r, s.c); visited.add(s.r*W + s.c);
    if (s.e && s.e !== 'bump') events.push({ t: i+1, e: s.e });
    frames.push({ r: s.r, c: s.c, k: s.k, seen: new Set(seen), visited: new Set(visited), events: [...events], bump: s.e === 'bump' });
  });
  return frames;
});

const boards = D.agents.map((a, i) => {
  const el = document.createElement('section'); el.className = 'board';
  el.innerHTML = `<header><h2>${a.name}</h2><div class="status" id="st${i}"></div></header>
    <canvas id="cv${i}" width="${W*CELL}" height="${H*CELL}" role="img" aria-label="${a.name} 的迷宫"></canvas>
    <div class="events" id="ev${i}"></div>`;
  document.getElementById('boards').appendChild(el);
  return { ctx: document.getElementById('cv'+i).getContext('2d'), st: document.getElementById('st'+i), ev: document.getElementById('ev'+i) };
});

const names = { key: '捡到钥匙', door: '开门', exit: '到达终点' };
function draw(i, t) {
  const fr = states[i][Math.min(t, states[i].length-1)], ctx = boards[i].ctx;
  const fog = document.getElementById('fog').checked, trail = document.getElementById('trail').checked;
  const gotKey = fr.events.some(e => e.e === 'key'), opened = fr.events.some(e => e.e === 'door');
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
    const ch = D.grid[r][c], x = c*CELL, y = r*CELL;
    let col = ch === '#' ? css('--wall') : css('--floor');
    if (trail && ch !== '#' && fr.visited.has(r*W+c)) col = css('--trail');
    ctx.fillStyle = col; ctx.fillRect(x, y, CELL, CELL);
    const mark = (color, inset) => { ctx.fillStyle = color; ctx.fillRect(x+inset, y+inset, CELL-2*inset, CELL-2*inset); };
    if (ch === 'S') mark(css('--start'), 7);
    if (ch === 'E') mark(css('--exit'), 5);
    if (ch === 'K' && !gotKey) mark(css('--key'), 6);
    if (ch === 'D') { mark(css('--door'), opened ? 9 : 3); }
    if (fog && !fr.seen.has(r*W+c)) { ctx.fillStyle = css('--fog'); ctx.fillRect(x, y, CELL, CELL); }
  }
  ctx.fillStyle = css('--agent');
  ctx.beginPath(); ctx.arc(fr.c*CELL + CELL/2, fr.r*CELL + CELL/2, CELL*0.34, 0, Math.PI*2); ctx.fill();
  if (fr.k) { ctx.fillStyle = css('--key'); ctx.beginPath(); ctx.arc(fr.c*CELL + CELL/2, fr.r*CELL + CELL/2, CELL*0.14, 0, Math.PI*2); ctx.fill(); }

  const a = D.agents[i], done = t >= a.steps.length;
  const tt = Math.min(t, a.steps.length);
  boards[i].st.innerHTML = `<b>${tt}</b> 步` + (done ? (a.success ? ` · <span class="ok">成功</span> · ${(a.steps.length / D.optimal).toFixed(2)}× 最优` : ' · <span class="bad">超时</span>') : (fr.bump ? ' · 撞墙' : ''));
  boards[i].ev.innerHTML = fr.events.map(e => `<span class="ev ${e.e}">${names[e.e]} @ ${e.t}</span>`).join('');
}

let t = 0, timer = null;
const scrub = document.getElementById('scrub'); scrub.max = maxLen;
function render() { boards.forEach((_, i) => draw(i, t)); scrub.value = t; document.getElementById('tlabel').textContent = `${t} 步`; }
function stop() { clearInterval(timer); timer = null; document.getElementById('play').textContent = '播放'; }
function play() {
  if (timer) return stop();
  if (t >= maxLen) t = 0;
  document.getElementById('play').textContent = '暂停';
  const tick = () => { t++; render(); if (t >= maxLen) stop(); };
  timer = setInterval(tick, 1000 / +document.getElementById('speed').value);
}
document.getElementById('play').onclick = play;
document.getElementById('stepb').onclick = () => { stop(); if (t < maxLen) t++; render(); };
document.getElementById('reset').onclick = () => { stop(); t = 0; render(); };
scrub.oninput = () => { stop(); t = +scrub.value; render(); };
document.getElementById('speed').oninput = () => { if (timer) { stop(); play(); } };
document.getElementById('fog').onchange = render; document.getElementById('trail').onchange = render;
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
