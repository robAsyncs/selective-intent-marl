# Selective intention communication in Overcooked-AI

Can two agents coordinate better if one tells the other what it plans to do?
In this pilot, sharing an intention every tick increased average team reward
by 2.9% on Cramped Room and 10.8% on Coordination Ring. The agents also got in
each other's way less often on both maps.

## What we built

[`pilot.py`](pilot.py) runs two scripted agents in Overcooked-AI. The agents
choose tasks with rules and use shortest-path search to reach the relevant
kitchen tile. The same controller runs in both conditions.

Each tick, one agent picks a task and target tile first. In the **silent**
condition it keeps that choice private. In **always-send** it tells its partner
what it plans to do, or sends an idle message if it has no task. If both want
the same exclusive target and the sender has a clear route and can get there
at least as quickly, the receiver picks another task or waits. The message
expires after that decision. Overcooked-AI applies both actions together, and
the agents swap sender and receiver roles on the next tick.

We ran 24 pairs of 400-tick episodes on each map. Within each pair, both
conditions used the same seed for starting orientations, route tie-breaking,
the first sender, and collision recovery choices.

## What happened

Values are averages per episode. Team reward is the simulator's raw reward,
and orders are completed soup deliveries. A blocked move is an attempted move
onto a floor tile that the simulator cancels because of a player collision.
We count a duplicate commitment when both agents choose the same exclusive
target on the same tick, such as the final ingredient slot in a pot.

| Layout | Condition | Team reward | Orders | Messages | Blocked moves | Duplicate commitments |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cramped Room | Silent | 199.17 | 9.96 | 0 | 80.04 | 47.33 |
| Cramped Room | Always-send | 205.00 | 10.25 | 400 | 68.38 | 31.33 |
| Coordination Ring | Silent | 139.17 | 6.96 | 0 | 197.75 | 25.33 |
| Coordination Ring | Always-send | 154.17 | 7.71 | 400 | 185.63 | 12.04 |

On Cramped Room, always-send earned more reward in 7 pairs, tied in 15, and
earned less in 2. On Coordination Ring, the counts were 12, 9, and 3.
The [saved traces](pilot-results.json) show decisions from the same state
with and without a message. On Cramped Room, the receiver waits after
learning that its partner is heading to start the soup. On Coordination Ring,
the receiver turns toward a different onion dispenser.

## Conflict-based sending

After choosing its task, the sender checks if the teammate could use the same
exclusive target given the visible kitchen state and what the teammate holds.
For example, two agents carrying onions can compete for the last slot in a
pot. The sender sends only if it has a route around the teammate and can reach
the target at least as quickly. The receiver chooses its intention and action
after this decision.

We evaluated the rule on the original 24 paired seeds per layout. The random
policy makes an independent send decision each tick. We calibrated its rate
on seeds 100–123 per layout and averaged eight random runs per evaluation seed.

| Layout | Rule reward | Rule messages | Random reward | Random messages | Paired reward gain (95% interval) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cramped Room | 205.00 | 43.58 | 200.52 | 47.08 | +4.48 (−0.21 to +9.69) |
| Coordination Ring | 154.17 | 30.50 | 139.27 | 27.04 | +14.90 (+5.10 to +25.31) |

The rule matched always-send reward on every paired seed. It used 89.1% fewer
messages on Cramped Room and 92.4% fewer on Coordination Ring. The paired
bootstrap interval for Cramped Room includes zero. The interval for Coordination
Ring favors the rule. See [heuristic-results.json](heuristic-results.json) for
per-seed metrics and the calibration rates.

## Learned sending

The learned policy uses the same send decision as the other policies. At a
potential conflict, its inputs are the sender's task, its distance to the
target, how many steps sooner it can get there than the teammate, and the
number of ticks left in the episode. The gate stays silent on the final tick,
when an eligible message cannot change team reward.

### Q update

The policy is fit with undiscounted, finite-horizon Q-learning over the send
gate's decisions ([`learned.py`](learned.py)). A decision is one eligible
conflict, described by features `x = (task, sender steps, lead)`, where
sender steps is capped at 4 and lead at 3. The time left in the episode is
`t = HORIZON − tick`. The two actions are `a ∈ {silent, send}`.

**Paired transitions.** Each training episode runs with random sending at an
exploration rate of 0, 0.5 or 1 per eligible conflict, cycling across seeds.
At up to four sampled conflicts, we rerun the episode with the opposite
choice at that tick only. Both runs use the same simulator seed and the same
gate random stream, so they stay identical before the decision. For each
branch we record:

- `r_k`: team reward `k` ticks after the decision, up to the next eligible
  conflict or the end of the episode.
- `Δ`: ticks until that next conflict.
- `x′`: the features at that conflict, or none if the episode ended first.

**Target.** For a branch observed from a decision with `t` ticks left:

```
y(t) = Σ_{k < min(Δ, t)} r_k  +  max_a′ Q(t − Δ, x′, a′)
```

The bootstrap term is zero when there is no `x′` or when `t − Δ ≤ 1`,
because the gate stays silent on the final tick. An unseen `(t, x′)` also
counts as zero. The max makes the update off-policy. Only rewards up to the
next conflict come from the exploration policy. Later rewards come from the
greedy fitted values.

**Update.** With `c` as the cost per message, and averaging over the
transitions `P(t, x)` pooled for that state:

```
Q(t, x, silent) = mean_{p ∈ P(t, x)} y_silent(t)
Q(t, x, send)   = mean_{p ∈ P(t, x)} y_send(t)  −  c
```

The cost is charged once, on the send branch, for the current message. No
message can be sent between eligible conflicts. Later messages pay their cost
through the bootstrapped `Q(·, send)` values.

**Fitting order and pooling.** `t` runs from 2 up to `HORIZON`. Every
bootstrap reads a strictly smaller `t − Δ`, so the targets are already
final when they are used. This makes one backward-induction sweep, not
an iterative fit. Data is sparse at each exact tick, so `P(t, x)` pools all
transitions with the same features whose recorded time left falls in the
same 25-tick band as `t`. Each one is then re-targeted with the exact `t`.
Rewards past `t` are dropped, and the bootstrap is read at `t − Δ`. A
transition recorded with fewer than `t` ticks left is used only if both
branches reached a next conflict. Otherwise the episode end would have cut
off rewards that a longer horizon would have counted.

**Acting.** The gate sends when `Q(t, x, send) > Q(t, x, silent)`. It stays
silent on ties, on unseen states, on ineligible ticks and on the final tick.
The same paired transitions are refit for each message cost, which gives one
policy per cost.

We trained separately on each layout using 512 seeds (1000–1511), with up
to four paired decisions sampled per seed. We evaluated all costs and
baselines on the same 48 held-out seeds (2000–2047) per layout. The random
baseline for the learned gate uses the same eligible conflict opportunities.
We calibrated its per-opportunity probability on seeds 100–123 to approximate
the learned gate's mean message count at each cost.

| Layout | Cost per message | Reward | Messages | Adjusted reward |
| --- | ---: | ---: | ---: | ---: |
| Cramped Room | 0.00 | 204.58 | 17.88 | 204.58 |
| Cramped Room | 0.10 | 204.17 | 16.06 | 202.56 |
| Cramped Room | 0.25 | 203.75 | 13.94 | 200.27 |
| Cramped Room | 0.50 | 204.17 | 11.35 | 198.49 |
| Coordination Ring | 0.00 | 146.25 | 6.29 | 146.25 |
| Coordination Ring | 0.10 | 146.25 | 6.04 | 145.65 |
| Coordination Ring | 0.25 | 147.92 | 5.48 | 146.55 |
| Coordination Ring | 0.50 | 150.00 | 4.63 | 147.69 |

At a cost of 0.50 on Coordination Ring, the learned policy kept 99.4% of
always-send reward. It earned 8.57 more adjusted reward points than the
random baseline with the same eligibility rules (paired 95% interval from
+2.54 to +14.24). Both policies averaged 5.54 messages on the calibration
seeds. On the held-out seeds, the learned policy averaged 4.63 messages and
the random baseline averaged 5.84. At a cost of 0.25, the paired gain was 5.52
(+0.04 to +10.90). The two lower-cost intervals include zero. On Cramped Room,
silence tied the learned policy at zero cost and scored higher at all three
positive costs. See
[learned-results.json](learned-results.json) for all baseline comparisons and
per-seed metrics.

## Run it locally

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). The simulator revision is
pinned in [`requirements.txt`](requirements.txt).

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python -m unittest
.venv/bin/python pilot.py --output /tmp/overcooked-pilot.json
cmp pilot-results.json /tmp/overcooked-pilot.json
.venv/bin/python experiments.py heuristic --output /tmp/overcooked-heuristic.json
cmp heuristic-results.json /tmp/overcooked-heuristic.json
.venv/bin/python experiments.py learned --output /tmp/overcooked-learned.json
cmp learned-results.json /tmp/overcooked-learned.json
```

[`pilot-results.json`](pilot-results.json) contains both conditions' metrics
for each seed and one controlled trace per map.
