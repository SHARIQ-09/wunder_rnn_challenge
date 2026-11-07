"""Learned gate over two frozen GRUs: does a per-step "how much to trust each model" network beat 50/50?

Models: v144 norm_ft and raw_ev_ema (our platform ensemble). Fit on 100 validation sequences,
score on 100 others (split by sequence), official mask. Gate output w in [0,1] per target:
pred = w * pA + (1 - w) * pB.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import FEATURE_COLUMNS
from src.evaluate import load_pytorch_model
from src.feature_sets import build_features, PRESSURE_GROUPS, N_EV_EMA

CKPTS = ["checkpoints_norm_ft/best_norm_ft.pt", "checkpoints_raw_ev_ema_s21/best.pt"]
N_SEQ, STEP = 200, 3
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = [(load_pytorch_model(Path(c), dev), torch.load(c, map_location="cpu").get("model_config", {}).get("feature_set", "v144"))
          for c in CKPTS]


@torch.no_grad()
def run_gru(m, x):
    """ScaledWunderGRU.forward that also returns the last layer's memory at every step."""
    z = (x - m.feat_mean) / m.feat_std
    if float(m.norm_clip) > 0:
        z = torch.clamp(z, -float(m.norm_clip), float(m.norm_clip))
    h = m.input_proj(z)
    for i, gru in enumerate(m.gru_layers):
        out, _ = gru(h)
        h = m.norms[i](out) if i == 0 else m.norms[i](out + h)
    return m.head(h)[0].float().cpu().numpy(), h[0].float().cpu().numpy()


# market-state inputs for the gate: activity bursts + liquidity/shock groups of the pressure set
P0 = 112 + N_EV_EMA + 3
STATE = list(range(P0 + PRESSURE_GROUPS["bursts"][0], P0 + PRESSURE_GROUPS["liquidity"][1]))

pf = pq.ParquetFile("datasets/valid.parquet")
ids = np.sort(np.random.default_rng(11).choice(pf.num_row_groups, N_SEQ, replace=False))
PA, PB, HM, S, Y, G = [], [], [], [], [], []
for g, i in enumerate(ids):
    d = pf.read_row_group(int(i), columns=FEATURE_COLUMNS + ["t0", "t1", "need_prediction", "is_scored"]).to_pandas()
    raw = d[FEATURE_COLUMNS].to_numpy(np.float32)
    rows = np.flatnonzero(d.need_prediction.values & d.is_scored.values)[::STEP]
    outs = [run_gru(m, torch.from_numpy(build_features(raw, fs)).unsqueeze(0).to(dev)) for m, fs in models]
    PA.append(outs[0][0][rows]); PB.append(outs[1][0][rows])
    HM.append(np.hstack([outs[0][1][rows], outs[1][1][rows]]).astype(np.float16))
    S.append(build_features(raw, "pressure")[rows][:, STATE])
    Y.append(np.clip(d[["t0", "t1"]].to_numpy(np.float32)[rows], -2, 2)); G.append(np.full(len(rows), g))
PA, PB, Y, G = np.vstack(PA), np.vstack(PB), np.vstack(Y), np.concatenate(G)
HM, S = np.vstack(HM).astype(np.float32), np.vstack(S)
fit, test = G < N_SEQ // 2, G >= N_SEQ // 2
inner = fit & (G >= int(N_SEQ * 0.4))          # last 20% of fit sequences: early stopping only
tr = fit & ~inner
print(f"rows fit {fit.sum()} test {test.sum()}", flush=True)


def wp(y, p):
    w = np.abs(y); my, mp = np.average(y, weights=w), np.average(p, weights=w)
    return np.sum(w * (y - my) * (p - mp)) / np.sqrt(np.sum(w * (y - my) ** 2) * np.sum(w * (p - mp) ** 2))


def report(name, p):
    s = [wp(Y[test, j], p[:, j]) for j in range(2)]
    print(f"{name:44s} t0 {s[0]:.4f}  t1 {s[1]:.4f}  WP {np.mean(s):.4f}", flush=True)
    return np.mean(s)


report("GRU A alone (v144 norm_ft)", PA[test]); report("GRU B alone (raw_ev_ema)", PB[test])
base = report("50/50 average (current platform best)", 0.5 * (PA[test] + PB[test]))
grid = np.linspace(0, 1, 21)
wbest = [grid[np.argmax([wp(Y[fit, j], a * PA[fit, j] + (1 - a) * PB[fit, j]) for a in grid])] for j in range(2)]
report(f"best fixed weights on A (t0 {wbest[0]:.2f}, t1 {wbest[1]:.2f})", np.column_stack([wbest[j] * PA[test, j] + (1 - wbest[j]) * PB[test, j] for j in range(2)]))


def torch_wp_loss(y, p):
    w = y.abs() + 1e-4
    mse = (w * (p - y) ** 2).sum(0) / w.sum(0)
    ym, pm = (w * y).sum(0) / w.sum(0), (w * p).sum(0) / w.sum(0)
    cov = (w * (y - ym) * (p - pm)).sum(0)
    corr = cov / torch.sqrt((w * (y - ym) ** 2).sum(0) * (w * (p - pm) ** 2).sum(0) + 1e-12)
    return (mse + 1.5 * (1 - corr)).mean()


def train_gate(X, name):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    Xs = {k: t((X[m] - mu) / sd) for k, m in (("tr", tr), ("ev", inner), ("te", test))}
    pa = {k: t(PA[m]) for k, m in (("tr", tr), ("ev", inner), ("te", test))}
    pb = {k: t(PB[m]) for k, m in (("tr", tr), ("ev", inner), ("te", test))}
    y = {k: t(Y[m]) for k, m in (("tr", tr), ("ev", inner))}
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.GELU(), nn.Linear(32, 2))
    nn.init.zeros_(net[2].weight); nn.init.zeros_(net[2].bias)        # starts exactly at 50/50
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    mix = lambda k: (lambda w: w * pa[k] + (1 - w) * pb[k])(torch.sigmoid(net(Xs[k])))
    best, best_state, bad = -1, None, 0
    for ep in range(400):
        perm = torch.randperm(len(Xs["tr"]))
        for b in perm.split(8192):
            opt.zero_grad()
            w = torch.sigmoid(net(Xs["tr"][b]))
            torch_wp_loss(y["tr"][b], w * pa["tr"][b] + (1 - w) * pb["tr"][b]).backward()
            opt.step()
        with torch.no_grad():
            p = mix("ev").numpy()
        s = np.mean([wp(Y[inner, j], p[:, j]) for j in range(2)])
        if s > best + 1e-5:
            best, best_state, bad = s, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 15:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        w = torch.sigmoid(net(Xs["te"])).numpy()
        p = mix("te").numpy()
    r = report(name, p)
    print(f"    gate weight on A: mean t0 {w[:, 0].mean():.2f} t1 {w[:, 1].mean():.2f} | spread (std) t0 {w[:, 0].std():.3f} t1 {w[:, 1].std():.3f} | epochs {ep + 1} | change vs 50/50 {r - base:+.4f}", flush=True)


train_gate(S, "gate from market state (bursts/liquidity/shocks)")
train_gate(HM, "gate from both GRU memories (256)")
train_gate(np.hstack([HM, S]), "gate from memories + market state")
