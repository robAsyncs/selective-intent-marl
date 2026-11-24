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
target, and how many steps sooner it can get there than the teammate.

During training, we replay a sampled decision with both choices, send and
stay silent, and measure the difference in team reward and message count
through the end of the episode. The paired rollouts share the simulator seed
and subsequent random draws. The model averages these differences for
similar decisions and sends when the estimated gain in team reward minus
the cost of extra messages is positive. The same rollout data is used to fit
decisions at each cost.

We trained separately on each layout using 512 seeds (1000–1511), with two
decisions sampled per seed. We evaluated all costs and baselines on the same
48 held-out seeds (2000–2047) per layout. Random sender rates were calibrated
on seeds 100–123.

| Layout | Cost per message | Reward | Messages | Adjusted reward |
| --- | ---: | ---: | ---: | ---: |
| Cramped Room | 0.00 | 202.92 | 32.02 | 202.92 |
| Cramped Room | 0.10 | 202.50 | 16.67 | 200.83 |
| Cramped Room | 0.25 | 201.67 | 5.10 | 200.39 |
| Cramped Room | 0.50 | 202.92 | 4.77 | 200.53 |
| Coordination Ring | 0.00 | 146.67 | 10.38 | 146.67 |
| Coordination Ring | 0.10 | 146.67 | 9.15 | 145.75 |
| Coordination Ring | 0.25 | 146.67 | 7.67 | 144.75 |
| Coordination Ring | 0.50 | 147.92 | 5.81 | 145.01 |

At a cost of 0.50 on Coordination Ring, the learned policy kept 98.1% of
always-send reward and sent 5.81 messages per episode. It beat its
rate-matched random baseline by 7.96 adjusted reward points (paired 95%
interval from +0.77 to +14.57). The other three intervals on Coordination
Ring include zero. On Cramped Room, silence scored higher than learned
sending on mean adjusted reward at every tested cost. See
[learned-results.json](learned-results.json) for all baseline comparisons and
per-seed metrics.

## Run it locally

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). The simulator revision is
pinned in [`requirements.txt`](requirements.txt).

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
.venv/bin/python pilot.py --output /tmp/overcooked-pilot.json
cmp pilot-results.json /tmp/overcooked-pilot.json
.venv/bin/python experiments.py heuristic --output /tmp/overcooked-heuristic.json
cmp heuristic-results.json /tmp/overcooked-heuristic.json
.venv/bin/python experiments.py learned --output /tmp/overcooked-learned.json
cmp learned-results.json /tmp/overcooked-learned.json
```

[`pilot-results.json`](pilot-results.json) contains both conditions' metrics
for each seed and one controlled trace per map.
