"""Fit finite-horizon action values for the send gate from simulator replay.

The send gate is treated as an undiscounted, finite-horizon decision problem
whose decisions are the eligible conflicts in an episode. The state is
``(t, x)``: ``t`` is the number of ticks left (``HORIZON - tick``) and ``x`` is
the ``Features`` tuple from ``send_features``. There are two actions, silent
and send. Q-values are stored as ``values[t][x] = (Q_silent, Q_send)``.

For each branch of a paired replay, the Bellman target is

    y(t) = sum of rewards r_k for k < min(elapsed, t)
           + max_a' Q(t - elapsed, x', a')

where ``x'`` is the next eligible conflict. The bootstrap term is zero when
the episode ended first, when ``t - elapsed <= 1`` (the gate is silent on the
final tick) or when ``(t - elapsed, x')`` was never fitted. The update
averages targets over the transitions pooled for ``(t, x)``:

    Q(t, x, silent) = mean(y_silent)
    Q(t, x, send)   = mean(y_send) - cost

Every bootstrap reads a smaller ``t``, so one sweep from ``t = 2`` up to
``HORIZON`` is exact backward induction over the pooled data. See the
"Q update" section of README.md.
"""

from collections import defaultdict
from dataclasses import dataclass
from random import Random

from pilot import HORIZON, SendPolicy, competition_distances, run_episode

TRAINING_SEEDS = range(1000, 1512)
PROBES_PER_EPISODE = 4
EXPLORATION_RATES = (0.0, 0.5, 1.0)
DEFAULT_COSTS = (0.0, 0.1, 0.25, 0.5)
TIME_BIN_WIDTH = 25

# (sender's task, sender's steps to the target capped at 4, how many steps
# sooner the sender arrives than the teammate capped at 3). The lead is never
# negative because competition_distances rejects targets the teammate is
# closer to.
Features = tuple[str, int, int]


def send_features(controller, state, sender, intent) -> Features | None:
    # An eligible message can only redirect a teammate with a conflicting task.
    # Neither that task nor the sender's can deliver a soup on the final tick.
    if intent is None or state.timestep >= HORIZON - 1:
        return None
    distances = competition_distances(controller, state, sender, intent)
    if distances is None:
        return None
    sender_steps, teammate_steps = distances
    return intent.task, min(sender_steps, 4), min(teammate_steps - sender_steps, 3)


@dataclass(frozen=True)
class Outcome:
    """Rewards and state observed up to the next eligible send decision.

    Covers one branch of a paired replay, starting at the probed tick: the
    transition ``(t, x, a) -> (r_0 .. r_{elapsed-1}, t - elapsed, x')``.
    """

    # Ticks from this decision to the next eligible one, or to the episode end.
    elapsed: int
    # Features x' at the next eligible decision, or None if the episode ended.
    successor: Features | None
    # Sparse (offset, reward) pairs. Offset 0 is the reward from the probed
    # tick's own transition.
    rewards: tuple[tuple[int, int], ...]

    def target(self, values, remaining):
        """Return the Bellman target y(t) for this branch with t = remaining.

        ``remaining`` can differ from the time left when the branch was
        recorded, because probes are pooled across a time band. The target is
        re-evaluated as if the decision happened with ``remaining`` ticks left.
        """
        # Rewards at offsets >= remaining would fall after the episode ends.
        reward = sum(amount for offset, amount in self.rewards if offset < remaining)
        next_remaining = remaining - self.elapsed
        # No bootstrap after the horizon, or on the final tick, where the gate
        # is always silent and Q is defined as zero.
        if self.successor is None or next_remaining <= 1:
            return reward
        # Greedy (off-policy) bootstrap: max over Q(t', x', silent) and
        # Q(t', x', send). Q(send) already has its message cost subtracted.
        # Unseen states count as (0, 0), the same value the gate acts on.
        return reward + max(values[next_remaining].get(self.successor, (0.0, 0.0)))


@dataclass(frozen=True)
class Probe:
    """One paired transition: both actions taken from the same state."""

    # Ticks left when the decision was recorded.
    remaining: int
    silence: Outcome
    send: Outcome

    def covers(self, remaining):
        """Whether this probe can supply a target for ``remaining`` ticks left.

        Fitting with no more time than was recorded only drops late rewards.
        Fitting with more time is valid only when both branches reached a next
        decision. Otherwise the recorded episode ended early, and the rewards
        a longer horizon would have seen are missing.
        """
        # A branch cut short by the training horizon supplies no later data.
        return self.remaining >= remaining or (
            self.silence.successor is not None and self.send.successor is not None
        )


@dataclass
class Rollout:
    """One training episode's eligible decisions and per-tick team rewards."""

    # tick -> (features, whether the gate sent). One sender per tick, so a
    # tick holds at most one decision.
    decisions: dict[int, tuple[Features, bool]]
    # rewards[tick] is the team reward from that tick's transition.
    rewards: list[int]

    def outcome(self, tick):
        """Cut the episode at the next eligible decision after ``tick``."""
        for end in range(tick + 1, len(self.rewards)):
            if end in self.decisions:
                successor, _ = self.decisions[end]
                break
        else:
            end, successor = len(self.rewards), None
        rewards = tuple(
            (offset, reward)
            for offset, reward in enumerate(self.rewards[tick:end])
            if reward
        )
        return Outcome(end - tick, successor, rewards)


def rollout(controller, seed, rate, override: tuple[int, bool] | None = None):
    """Run one episode with epsilon-style random sending at eligible conflicts.

    ``rate`` is the send probability at each eligible decision. ``override``
    forces one decision ``(tick, sent)``. Replaying a seed with the opposite
    choice at one tick gives the paired branch.
    """
    rng = Random(f"training-rollout:{seed}")
    trajectory = Rollout({}, [])

    def send(controller, state, sender, intent):
        features = send_features(controller, state, sender, intent)
        # Draw on every call, eligible or not, so the paired branches use the
        # same random numbers at the same ticks after they diverge.
        draw = rng.random()
        if features is None:
            return False

        tick = state.timestep
        decision = draw < rate
        if override is not None and tick == override[0]:
            decision = override[1]
        trajectory.decisions[tick] = features, decision
        return decision

    run_episode(controller, seed, send, observe_reward=trajectory.rewards.append)
    return trajectory


def time_bin(remaining):
    """Band of TIME_BIN_WIDTH ticks used to pool probes across nearby times."""
    return (remaining - 1) // TIME_BIN_WIDTH


def collect_send_probes(controller):
    """Replay both actions at the same states using paired simulator seeds.

    Returns probes keyed by ``(time band, features)``. Exploration rates cycle
    across seeds so the data includes histories with no messages, some
    messages and all messages around the probed decision.
    """
    probes = defaultdict(list)
    rng = Random(0)
    for index, seed in enumerate(TRAINING_SEEDS):
        rate = EXPLORATION_RATES[index % len(EXPLORATION_RATES)]
        original = rollout(controller, seed, rate)
        for tick in rng.sample(
            list(original.decisions), min(PROBES_PER_EPISODE, len(original.decisions))
        ):
            features, sent = original.decisions[tick]
            # Same seed and gate stream, opposite action at this tick only.
            # Later decisions still follow the exploration rate.
            alternate = rollout(controller, seed, rate, override=(tick, not sent))
            silence, send = (alternate, original) if sent else (original, alternate)
            remaining = HORIZON - tick
            probes[time_bin(remaining), features].append(
                Probe(remaining, silence.outcome(tick), send.outcome(tick))
            )
    return probes


def fit_send_policy(probes, cost: float) -> SendPolicy:
    """Fit Q with undiscounted Bellman targets, starting at the end of the episode.

    ``values[t][x]`` holds ``(Q_silent, Q_send)`` for ``t`` ticks left. ``t = 1``
    (the final tick) stays empty because the gate never sends there.
    """
    values = [{} for _ in range(HORIZON + 1)]
    # Backward induction: a target at t bootstraps from t - elapsed < t, which
    # was already fitted earlier in this loop.
    for remaining in range(2, HORIZON + 1):
        for (band, features), samples in probes.items():
            # Pool probes from the same band, then re-target each one at the
            # exact remaining time.
            if band != time_bin(remaining):
                continue
            usable = [probe for probe in samples if probe.covers(remaining)]
            if usable:
                # Silence is forced between eligible decisions. Only the send
                # branch pays for the message at the current decision.
                values[remaining][features] = (
                    sum(probe.silence.target(values, remaining) for probe in usable)
                    / len(usable),
                    sum(probe.send.target(values, remaining) for probe in usable)
                    / len(usable) - cost,
                )

    def send(controller, state, sender, intent):
        features = send_features(controller, state, sender, intent)
        if features is None:
            return False
        q_silence, q_send = values[HORIZON - state.timestep].get(features, (0.0, 0.0))
        # Greedy action. Ties and unseen states stay silent.
        return q_send > q_silence

    return send
