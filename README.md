# Pump.fun top-5 forecaster

A calibrated Monte-Carlo forecast for make.fun's `pumpfun-top5-forecast-01`:
which 5 pump.fun coins launched since the opening will be largest by market
cap at seven hourly reads a day later, with honest probabilities per coin and
place, and a probability distribution over market-cap size bins per place.

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads `/input/task.json` and `/input/history.ndjson` and writes
`/output/forecast.json`.

## Model

For every candidate coin (strong at the last snapshot, recently near the
top, or rising), its future log10 market cap at the answer horizon is
modelled as Normal(mu, sigma):

- **mu** — last log-cap plus a shrunk drift times the horizon. The drift is
  the coin's own recent log-cap slope (least squares over its last few
  snapshots), shrunk toward the top coins' median slope, with an extra fade
  for coins already several hours old; the total move is clamped.
- **sigma** — a horizon-scaled base volatility times the coin's own recent
  per-step volatility relative to the field median.
- **Phantom newcomers** — coins launched after the history ends are drawn as
  a few pseudo-coins whose best cap sits near the current place-1 cap,
  lower the older the launch window already is. Whatever probability they
  (and non-candidate coins) take in the simulation is exactly the mass the
  forecast leaves unnamed, which is how the scorer pays for newcomers.

A fixed-seed simulation (SplitMix64, 8,000 draws) ranks every draw. Place
probabilities are the simulated place frequencies, pruned where a named row
would pay less than the implied unnamed share. The top-5 guess orders coins
by simulated top-5 frequency. Each place's size distribution is a normal
fitted to the simulation's own place-cap moments (sd shrunk, since an order
statistic is steadier than a single coin), mixed with the simulated
histogram for shape, over all 82 bins.

## Measured on the kit rehearsal

Backtested inside the rehearsal history with an exact port of the scorer
(for each cut, history up to that snapshot, answers after it), against the
house persistence baseline scored the same way:

| cut | this forecaster | persistence |
|-----|----------------:|------------:|
| +10 min history | −39.47 | −59.11 |
| +20 min history | −38.03 | −49.88 |
| +30 min history | −33.64 | −33.48 |
| +40 min history | −34.96 | −29.87 |

Mean −36.52 vs −43.08 (higher is better). The house baseline's published
score on the full practice forecast is −40.408345.
