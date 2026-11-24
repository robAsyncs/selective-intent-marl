"""Fit a send policy from paired Monte Carlo estimates of action values."""

from collections import defaultdict
from dataclasses import dataclass
from random import Random

from pilot import Metrics, SendPolicy, competition_distances, run_episode

TRAINING_SEEDS = range(1000, 1512)
PROBES_PER_EPISODE = 2
ROLLOUT_RATES = (0.0, 0.5, 1.0)
DEFAULT_COSTS = (0.0, 0.1, 0.25, 0.5)
PRIOR_WEIGHT = 12

Features = tuple[str, int, int]


def send_features(controller, state, sender, intent) -> Features | None:
    if intent is None:
        return None
    distances = competition_distances(controller, state, sender, intent)
    if distances is None:
        return None
    sender_steps, teammate_steps = distances
    # Group distant routes and large leads so similar decisions share data.
    return intent.task, min(sender_steps, 4), min(teammate_steps - sender_steps, 3)


@dataclass
class RolloutTotals:
    samples: int = 0
    reward_delta: int = 0
    message_delta: int = 0

    def record(self, send: Metrics, stay_silent: Metrics):
        self.samples += 1
        self.reward_delta += send.team_return - stay_silent.team_return
        self.message_delta += send.messages_sent - stay_silent.messages_sent

    def adjusted_delta(self, cost):
        return self.reward_delta - cost * self.message_delta


class SendValues:
    def __init__(self):
        self.by_task: dict[str, RolloutTotals] = defaultdict(RolloutTotals)
        self.by_features: dict[Features, RolloutTotals] = defaultdict(RolloutTotals)

    def record(self, features: Features, send: Metrics, stay_silent: Metrics):
        self.by_task[features[0]].record(send, stay_silent)
        self.by_features[features].record(send, stay_silent)

    def advantage(self, features: Features, cost: float) -> float:
        # Shrink a sparse bin toward the estimate from other bins for this task.
        local = self.by_features.get(features, RolloutTotals())
        task = self.by_task.get(features[0], RolloutTotals())
        other_samples = task.samples - local.samples
        other_delta = task.adjusted_delta(cost) - local.adjusted_delta(cost)
        prior = other_delta / (other_samples + PRIOR_WEIGHT)
        return (local.adjusted_delta(cost) + PRIOR_WEIGHT * prior) / (
            local.samples + PRIOR_WEIGHT
        )

    def policy(self, cost: float) -> SendPolicy:
        def send(controller, state, sender, intent):
            features = send_features(controller, state, sender, intent)
            return features is not None and self.advantage(features, cost) > 0

        return send


def rollout(controller, seed, rate, override: tuple[int, bool] | None = None):
    rng = Random(f"training-rollout:{seed}")
    tick = 0
    opportunities = []

    def send(controller, state, sender, intent):
        nonlocal tick
        features = send_features(controller, state, sender, intent)
        # Use the same random draw at each tick even when the two paths diverge.
        draw = rng.random()
        decision = features is not None and draw < rate
        if override is None:
            if features is not None:
                opportunities.append((tick, features, decision))
        elif tick == override[0]:
            decision = override[1]
        tick += 1
        return decision

    metrics, _ = run_episode(controller, seed, send)
    return metrics, opportunities


def fit_send_values(controller):
    """Estimate Q(send) - Q(silence) under a mixture of continuation policies."""
    values = SendValues()
    rng = Random(0)
    for index, seed in enumerate(TRAINING_SEEDS):
        rate = ROLLOUT_RATES[index % len(ROLLOUT_RATES)]
        original, opportunities = rollout(controller, seed, rate)
        for tick, features, sent in rng.sample(
            opportunities, min(PROBES_PER_EPISODE, len(opportunities))
        ):
            alternate, _ = rollout(controller, seed, rate, override=(tick, not sent))
            send, stay_silent = (original, alternate) if sent else (alternate, original)
            # The two rollouts have the same prefix, so full-episode differences
            # give the difference in reward and message count from this tick on.
            values.record(features, send, stay_silent)
    return values
