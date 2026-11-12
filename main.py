#!/usr/bin/env python3
"""pumpfun-top5-forecast-01: calibrated Monte-Carlo forecaster (Python 3.13, stdlib only).

Reads /input/task.json + /input/history.ndjson (10-minute market-cap snapshots of
every pump.fun coin launched at/after the opening) and writes /output/forecast.json.

Model: for each candidate coin, its future log10 market cap (SOL) at the answer
horizon is Normal(mu, sigma):
  mu    = last log-cap + shrunk per-coin drift * horizon (drift from the coin's
          recent log-cap slope, shrunk toward the top coins' median slope, with
          an extra fade for old coins), total move clamped;
  sigma = horizon-scaled base volatility times the coin's own recent volatility.
Phantom newcomers (coins launched after the history ends) are drawn as a few
extra pseudo-coins whose best cap sits near the current place-1 cap, lower the
older the launch window already is; they absorb exactly the probability mass
that scoring leaves to the unnamed share. A fixed-seed simulation ranks all
draws; place probabilities and per-place size-bin histograms come straight
from the simulated rankings, lightly smoothed. Deterministic: same history,
same forecast. No network, no model calls.
"""

import json
import math
import os
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

DRAWS = 8000
SEED = 987654321


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


class Rng:
    """SplitMix64: tiny deterministic PRNG, identical on every platform."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next_u64() >> 11) / float(1 << 53)

    def normal(self):
        # Box-Muller on two uniforms in (0,1).
        u1 = 1.0 - self.uniform()
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def load_input():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise ValueError("history holds no ok mark")
    return task, marks


def coin_series(marks):
    """mint -> list of (mark_index, log10 cap SOL, row) plus static info."""
    series = {}
    info = {}
    for idx, m in enumerate(marks):
        for r in m["rows"]:
            mint = r["mint"]
            lc = math.log10(int(r["market_cap_lamports"])) - 9.0
            series.setdefault(mint, []).append((idx, lc))
            info[mint] = r
    return series, info


def main():
    task, marks = load_input()
    places = int(task["places"])
    edges = task["size"]["edges"]
    nbins = len(edges) + 1
    t0 = parse_ts(task["t0"])
    last = marks[-1]
    last_t = parse_ts(last["at"])
    reads = [parse_ts(r) for r in task["reads"]]
    horizon_h = max(0.05, (sum(reads) / len(reads) - last_t) / 3600.0)
    window_age_h = max(0.05, (last_t - t0) / 3600.0)
    mark_times = [parse_ts(m["at"]) for m in marks]

    series, info = coin_series(marks)
    last_rows = sorted(last["rows"], key=lambda r: r["rank"])
    if len(last_rows) < places:
        raise ValueError("last mark ranks fewer coins than places")
    last_rank = {r["mint"]: i + 1 for i, r in enumerate(last_rows)}
    place_caps = [
        math.log10(int(last_rows[k]["market_cap_lamports"])) - 9.0 for k in range(places)
    ]

    # Per-place cap trend (log10 per hour) across the history, shrunk hard:
    # the whole field's caps trend together, and place caps follow it.
    def series_slope(xs, ys):
        n = len(xs)
        if n < 2:
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den <= 0:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den * 3600.0

    place_slopes = []
    for k in range(places):
        xs, ys = [], []
        for mi, m in enumerate(marks):
            rows = sorted(m["rows"], key=lambda r: r["rank"])
            if len(rows) > k:
                xs.append(mark_times[mi])
                ys.append(math.log10(int(rows[k]["market_cap_lamports"])) - 9.0)
        place_slopes.append(max(-0.8, min(0.8, 0.5 * series_slope(xs, ys))))
    market_slope = sorted(place_slopes)[len(place_slopes) // 2] if place_slopes else 0.0

    # ---- per-coin features -------------------------------------------------
    def slope_per_hour(mint):
        pts = series[mint]
        if len(pts) < 2:
            return None
        # Use up to the last 7 appearances (~1h), least squares on time.
        pts = pts[-7:]
        xs = [mark_times[i] for i, _ in pts]
        ys = [lc for _, lc in pts]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den <= 0:
            return None
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den * 3600.0

    def vol_per_step(mint):
        pts = series[mint]
        if len(pts) < 3:
            return None
        diffs = [pts[i + 1][1] - pts[i][1] for i in range(len(pts) - 1)]
        diffs = diffs[-8:]
        mean = sum(diffs) / len(diffs)
        return math.sqrt(sum((d - mean) ** 2 for d in diffs) / len(diffs))

    top_mints = [r["mint"] for r in last_rows[:25]]
    slopes = {m: slope_per_hour(m) for m in top_mints}
    known = sorted(s for s in slopes.values() if s is not None)
    med_slope = known[len(known) // 2] if known else 0.0
    vols = {m: vol_per_step(m) for m in series}
    known_vols = sorted(v for v in vols.values() if v is not None)
    med_vol = known_vols[len(known_vols) // 2] if known_vols else 0.15

    # Candidate set: strong at the last mark, recently near the top, or rising.
    cand = {}
    for r in last_rows[:100]:
        cand[r["mint"]] = True
    for mint, pts in series.items():
        recent = [lc for i, lc in pts if i >= len(marks) - 3]
        if recent and max(recent) >= place_caps[-1] - 0.7:
            cand[mint] = True
    for mint in list(cand):
        pts = series[mint]
        last_lc = pts[-1][1] if pts[-1][0] == len(marks) - 1 else None
        if last_lc is None:
            # Not ranked at the last mark: keep only if its recent peak is close.
            if max(lc for _, lc in pts) < place_caps[-1] - 0.15:
                del cand[mint]

    coins = []
    for mint in cand:
        pts = series[mint]
        last_lc = None
        for i, lc in reversed(pts):
            if i == len(marks) - 1:
                last_lc = lc
                break
        if last_lc is None:
            last_lc = pts[-1][1] - 0.35 * (len(marks) - 1 - pts[-1][0])
        s = slope_per_hour(mint)
        if s is None:
            s = med_slope
        age_h = (last_t - parse_ts(info[mint]["created_at"])) / 3600.0
        drift = 0.55 * s + 0.45 * med_slope
        if age_h > 4.0:
            drift -= min(0.12, 0.02 * (age_h - 4.0))
        drift = max(-0.30, min(0.30, drift))
        move = max(-1.15, min(1.15, drift * horizon_h))
        mu = last_lc + move
        v = vols[mint] if vols[mint] is not None else med_vol
        vol_scale = 0.75 + 0.6 * min(2.0, v / med_vol) if med_vol > 0 else 1.0
        sigma = (0.34 + 0.30 * math.sqrt(horizon_h)) * vol_scale
        sigma = max(0.30, min(1.45, sigma))
        # Crash hazard: with probability stay_p the coin simply holds near
        # its cap (small noise, partial drift); otherwise it takes the full
        # diffuse move above. Graduated, complete coins are stickier. The
        # stay probability decays with the horizon, so at long horizons the
        # model is nearly pure diffusion again.
        tau = 2.2
        if info[mint].get("venue") == "pool":
            tau *= 1.5
        if info[mint].get("complete"):
            tau *= 1.25
        stay_p = math.exp(-horizon_h / tau)
        mu_stay = last_lc + 0.3 * move
        sig_stay = 0.16 + 0.05 * math.sqrt(horizon_h)
        coins.append((mint, mu, sigma, last_rank.get(mint, 999), stay_p, mu_stay, sig_stay))
    # Keep the strongest candidates by mu so the simulation stays fast.
    coins.sort(key=lambda c: -c[1])
    coins = coins[:56]

    # ---- phantom newcomers ---------------------------------------------------
    delta_new = 0.45 - 0.035 * window_age_h - 0.10 * math.log10(1.0 + horizon_h)
    delta_new = max(-0.15, min(0.45, delta_new))
    phantoms = []
    for j in range(3):
        phantoms.append((place_caps[0] + delta_new - 0.28 * j, 0.50))

    # ---- simulation ----------------------------------------------------------
    rng = Rng(SEED)
    n_all = len(coins) + len(phantoms)
    place_hits = [[0] * n_all for _ in range(places)]
    bin_hits = [[0] * nbins for _ in range(places)]
    cap_sum = [0.0] * places
    cap_sq = [0.0] * places
    top5_hits = [0] * n_all
    exp_rank = [0.0] * n_all
    mus = [c[1] for c in coins] + [p[0] for p in phantoms]
    sigs = [c[2] for c in coins] + [p[1] for p in phantoms]
    stays = [c[4] for c in coins] + [0.0] * len(phantoms)
    mus_stay = [c[5] for c in coins] + [p[0] for p in phantoms]
    sigs_stay = [c[6] for c in coins] + [p[1] for p in phantoms]

    def bin_index(x):
        b = 0
        for e in edges:
            if x >= e:
                b += 1
            else:
                break
        return b

    for _ in range(DRAWS):
        draws = []
        for i in range(n_all):
            if stays[i] > 0.0 and rng.uniform() < stays[i]:
                draws.append((mus_stay[i] + sigs_stay[i] * rng.normal(), i))
            else:
                draws.append((mus[i] + sigs[i] * rng.normal(), i))
        draws.sort(key=lambda t: -t[0])
        for k in range(places):
            val, idx = draws[k]
            place_hits[k][idx] += 1
        # Size moments come from real coins only: the k-th real coin's cap in
        # this draw. Phantom caps are a modelling device for *who* holds a
        # place, not a measurement of how large places get; the field's cap
        # trend (below) carries the level instead.
        seen = 0
        for val, idx in draws:
            if idx < len(coins):
                bin_hits[seen][bin_index(val)] += 1
                cap_sum[seen] += val
                cap_sq[seen] += val * val
                seen += 1
                if seen == places:
                    break
        for r_idx, (val, idx) in enumerate(draws[:places]):
            top5_hits[idx] += 1
            exp_rank[idx] += r_idx + 1

    # ---- probabilities -------------------------------------------------------
    probabilities = []
    for k in range(places):
        rows = []
        for ci, c in enumerate(coins):
            p = place_hits[k][ci] / DRAWS
            if p >= 0.004:
                rows.append((c[0], round(p, 6)))
        # Prune rows that pay less than the implied unnamed share would.
        for _ in range(3):
            named = len(rows)
            ssum = sum(p for _, p in rows)
            share = (1.0 - ssum) / max(1, 100 - named)
            keep = [(m, p) for m, p in rows if p >= share * 0.9]
            if len(keep) == len(rows):
                break
            rows = keep
        for mint, p in rows:
            probabilities.append({"mint": mint, "place": k + 1, "p": p})

    order = sorted(
        range(len(coins)),
        key=lambda i: (-top5_hits[i], exp_rank[i] / max(1, top5_hits[i])),
    )
    top5 = [coins[i][0] for i in order[:places]]
    if len(top5) < places:  # extremely thin history: fall back to last chart
        for r in last_rows:
            if r["mint"] not in top5:
                top5.append(r["mint"])
            if len(top5) == places:
                break

    # ---- sizes ---------------------------------------------------------------
    # The place cap's distribution: a normal centred halfway between the
    # simulation's real-coin place-cap mean and the place's current cap
    # extrapolated by its own trend across the history (the field's caps
    # trend hard, especially while the window is young), with a wide,
    # sim-scaled sd; mixed with the simulated histogram for shape and tails.
    sizes = []
    for k in range(places):
        hist = [bin_hits[k][b] / DRAWS for b in range(nbins)]
        sim_mean = cap_sum[k] / DRAWS
        sim_var = max(0.0, cap_sq[k] / DRAWS - sim_mean * sim_mean)
        sim_sd = math.sqrt(sim_var)
        trend_cap = place_caps[k] + place_slopes[k] * horizon_h
        mean_k = 0.5 * sim_mean + 0.5 * trend_cap
        sd_floor = 0.50 if window_age_h < 3.0 else 0.42
        sd_k = max(sd_floor, min(1.0, 0.55 * sim_sd))
        # While the field is still expanding fast, part of the mass rides a
        # second, higher centre: the frenzy continuing, not stopping at the cut.
        exp_w = 0.30 if market_slope > 0.15 else 0.0
        mean_hi = mean_k + 1.2 * market_slope * horizon_h

        def cdf(x, mu, sd):
            return 0.5 * (1.0 + math.erf((x - mu) / (sd * math.sqrt(2.0))))

        mix = []
        for b in range(nbins):
            lo = edges[b - 1] if b > 0 else mean_k - 8 * sd_k
            hi = edges[b] if b < nbins - 1 else mean_k + 8 * sd_k
            np_ = (1.0 - exp_w) * max(0.0, cdf(hi, mean_k, sd_k) - cdf(lo, mean_k, sd_k))
            if exp_w > 0.0:
                np_ += exp_w * max(0.0, cdf(hi, mean_hi, sd_k) - cdf(lo, mean_hi, sd_k))
            hp = 0.5 * hist[b]
            if b > 0:
                hp += 0.25 * hist[b - 1]
            if b < nbins - 1:
                hp += 0.25 * hist[b + 1]
            mix.append(0.85 * np_ + 0.15 * hp)
        # Name every bin: the cap must land somewhere, and an unnamed bin
        # would only receive the thin shared remainder.
        total = sum(mix)
        raw = [max(mix[b] / total, 0.0008) for b in range(nbins)]
        rtot = sum(raw)
        for b in range(nbins):
            p = round(raw[b] / rtot * 0.9995, 6)
            if p > 0:
                sizes.append({"place": k + 1, "bin": b, "p": p})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
