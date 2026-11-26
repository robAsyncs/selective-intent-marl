"""Evaluate selective send policies on paired Overcooked scenarios."""

import argparse
import json
from dataclasses import asdict
from functools import cache
from math import isfinite
from pathlib import Path
from random import Random
from statistics import mean

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from learned import (
    DEFAULT_COSTS,
    EXPLORATION_RATES,
    PROBES_PER_EPISODE,
    TIME_BIN_WIDTH,
    TRAINING_SEEDS,
    collect_send_probes,
    fit_send_policy,
    send_features,
)
from pilot import (
    HORIZON,
    LAYOUTS,
    SCENARIOS,
    Controller,
    always_send,
    conflict_rule,
    run_episode,
    silent,
)

# Seed ranges never overlap: calibration 100-123, training 1000-1511 (see
# learned.py), held-out evaluation 2000-2047. The heuristic experiment
# evaluates on the pilot seeds 0-23.
CALIBRATION_SEEDS = range(100, 124)
# Random gates are averaged over this many runs per seed to reduce their noise.
RANDOM_REPEATS = 8
# Bisection steps when matching a random gate's message budget.
CALIBRATION_STEPS = 8
HELD_OUT_SEEDS = range(2000, 2048)
BOOTSTRAP_RESAMPLES = 2000
# Paired adjusted-reward comparisons reported at every message cost.
SWEEP_COMPARISONS = (
    ("learned", "silent"),
    ("learned", "always_send"),
    ("learned", "conflict_rule"),
    ("learned", "random_matched_to_learned"),
    ("conflict_rule", "silent"),
    ("conflict_rule", "always_send"),
    ("conflict_rule", "random_matched"),
    ("always_send", "silent"),
)


def iqm(values):
    """Interquartile mean: the mean after trimming 25% from each end.

    Matches scipy.stats.trim_mean(values, 0.25), which the rliable library
    uses: floor(n / 4) values are dropped from each end.
    """
    ordered = sorted(values)
    cut = len(ordered) // 4
    return mean(ordered[cut : len(ordered) - cut])


def percentile_interval(resamples):
    """Return the 2.5th and 97.5th percentiles of bootstrap resamples."""
    ordered = sorted(resamples)
    count = len(ordered)
    return [ordered[round(0.025 * count)], ordered[round(0.975 * count) - 1]]


def bootstrap_seeds(seeds, statistic):
    """Percentile bootstrap over seeds. ``statistic`` maps a seed list to a float."""
    rng = Random(0)
    return percentile_interval(
        [statistic(rng.choices(seeds, k=len(seeds))) for _ in range(BOOTSTRAP_RESAMPLES)]
    )


def summarize(values_by_seed):
    """IQM and mean over seeds, with a bootstrap 95% interval for the IQM."""
    seeds = sorted(values_by_seed)
    return {
        "iqm": iqm(values_by_seed.values()),
        "iqm_95_percent_interval": bootstrap_seeds(
            seeds, lambda sample: iqm(values_by_seed[seed] for seed in sample)
        ),
        "mean": mean(values_by_seed.values()),
    }


def adjusted_returns(condition, cost):
    """Per-seed team reward minus the message cost times messages sent."""
    return {
        episode["seed"]: episode["team_return"] - cost * episode["messages_sent"]
        for episode in condition["episodes"]
    }


def random_policy(rate, seed, repeat, *, eligible_only=False):
    """Send with probability ``rate`` per tick, or per eligible tick.

    With ``eligible_only``, only ticks where the learned gate could act count
    as opportunities. That gives a random baseline with the same eligibility
    as the learned gate.
    """
    # Draw independently of the simulator RNG so each condition shares setup
    # and collision recovery draws with its paired scenarios.
    rng = Random(f"random-send:{seed}:{repeat}")

    def send(controller, state, sender, intent):
        # The draw comes first and happens every tick, so the random stream
        # does not depend on which ticks are eligible.
        return rng.random() < rate and (
            not eligible_only or send_features(controller, state, sender, intent) is not None
        )

    return send


def eligible_random_policy(rate):
    """Policy factory for a random gate limited to learned-gate opportunities."""

    def make_policy(seed, repeat):
        return random_policy(rate, seed, repeat, eligible_only=True)

    return make_policy


def evaluate(controller, seeds, make_policy, repeats=1):
    """Run a condition on each seed and summarize its metrics.

    ``make_policy(seed, repeat)`` builds a fresh send policy for each run.
    With several repeats, each seed's metrics are averaged first, so every
    seed counts once in the summaries and in paired comparisons.
    """
    episodes = []
    for seed in seeds:
        trials = []
        for repeat in range(repeats):
            metrics, _ = run_episode(controller, seed, make_policy(seed, repeat))
            trials.append(asdict(metrics))
        episode = {
            metric: mean(trial[metric] for trial in trials) for metric in trials[0]
        }
        episodes.append({"seed": seed, **episode})
    return {
        "means": {
            metric: mean(episode[metric] for episode in episodes) for metric in trials[0]
        },
        "iqms": {metric: iqm(episode[metric] for episode in episodes) for metric in trials[0]},
        "episodes": episodes,
    }


def compare_rewards(policy, reference, message_cost=0):
    """Compare communication-adjusted reward between two conditions, paired by seed.

    The headline statistic is the difference in IQM between the two
    conditions. Each bootstrap resample draws seeds with replacement and uses
    the same seeds for both conditions, so the interval keeps the pairing.
    The mean of the per-seed differences is kept for comparison.
    """
    for condition in (policy, reference):
        seeds = [episode["seed"] for episode in condition["episodes"]]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Reward comparisons require one episode per seed per condition")
    trial = adjusted_returns(policy, message_cost)
    baseline = adjusted_returns(reference, message_cost)
    if not trial or trial.keys() != baseline.keys():
        raise ValueError("Reward comparisons require the same nonempty set of seeds")

    def iqm_difference(sample):
        return iqm(trial[seed] for seed in sample) - iqm(baseline[seed] for seed in sample)

    seeds = sorted(trial)
    deltas = [trial[seed] - baseline[seed] for seed in seeds]
    return {
        "iqm_difference": iqm_difference(seeds),
        "iqm_difference_95_percent_interval": bootstrap_seeds(seeds, iqm_difference),
        "mean_difference": mean(deltas),
        "mean_difference_95_percent_interval": bootstrap_seeds(
            list(range(len(deltas))), lambda sample: mean(deltas[i] for i in sample)
        ),
    }


def evaluate_baselines(controller, seeds, calibration_seeds=CALIBRATION_SEEDS):
    """Evaluate the fixed gates, plus a random gate matched to the heuristic.

    The random gate's per-tick rate is the heuristic's mean messages per tick
    on the calibration seeds, so the two spend about the same budget.
    """
    calibration = evaluate(controller, calibration_seeds, lambda *_: conflict_rule)
    rate = calibration["means"]["messages_sent"] / HORIZON
    return rate, {
        "silent": evaluate(controller, seeds, lambda *_: silent),
        "always_send": evaluate(controller, seeds, lambda *_: always_send),
        "conflict_rule": evaluate(controller, seeds, lambda *_: conflict_rule),
        "random_matched": evaluate(
            controller,
            seeds,
            lambda seed, repeat: random_policy(rate, seed, repeat),
            repeats=RANDOM_REPEATS,
        ),
    }


def match_message_budget(messages_at_rate, target):
    """Find a per-opportunity rate close to the target calibration volume.

    Bisects on the rate, assuming messages rise with the rate. Returns the
    tried rate whose message count was closest to ``target``, along with that
    count. ``messages_at_rate`` is cached by the caller, so repeated rates
    cost nothing.
    """
    if target == 0:
        return 0.0, 0.0
    lower, upper = 0.0, 1.0
    full_count = messages_at_rate(upper)
    if full_count == target:
        return upper, full_count
    candidates = [(lower, 0.0), (upper, full_count)]
    for _ in range(CALIBRATION_STEPS):
        rate = (lower + upper) / 2
        message_count = messages_at_rate(rate)
        candidates.append((rate, message_count))
        if message_count == target:
            break
        if message_count < target:
            lower = rate
        else:
            upper = rate
    return min(candidates, key=lambda candidate: abs(candidate[1] - target))


def run_heuristic():
    """Compare the heuristic gate with its matched random gate on the pilot seeds."""
    results = {
        "horizon": HORIZON,
        "evaluation_seeds": list(range(SCENARIOS)),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "random_repeats_per_seed": RANDOM_REPEATS,
        "layouts": {},
    }
    for layout in LAYOUTS:
        controller = Controller(OvercookedGridworld.from_layout_name(layout))
        rate, conditions = evaluate_baselines(controller, range(SCENARIOS))
        rule = conditions["conflict_rule"]
        full = conditions["always_send"]
        results["layouts"][layout] = {
            "calibrated_random_rate": rate,
            "conditions": conditions,
            "always_send_reward_retained": (
                rule["means"]["team_return"] / full["means"]["team_return"]
            ),
            "team_return": {
                name: summarize(adjusted_returns(condition, 0))
                for name, condition in conditions.items()
            },
            "rule_vs_random_reward": compare_rewards(rule, conditions["random_matched"]),
        }
    return results


def run_learned(costs=DEFAULT_COSTS):
    """Fit the learned gate at each cost and score every condition on held-out seeds.

    For each layout: collect paired training transitions once, evaluate the
    fixed baselines once, then for each cost fit a policy, match a random gate
    to its message budget on the calibration seeds, and compare all
    conditions on the held-out seeds.
    """
    results = {
        "horizon": HORIZON,
        "training_seeds": list(TRAINING_SEEDS),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "evaluation_seeds": list(HELD_OUT_SEEDS),
        "probes_per_training_episode": PROBES_PER_EPISODE,
        "training_exploration_rates": EXPLORATION_RATES,
        "training_time_bin_width": TIME_BIN_WIDTH,
        "random_repeats_per_seed": RANDOM_REPEATS,
        "random_matched_to_learned_eligibility": "same_as_learned",
        "layouts": {},
    }
    for layout in LAYOUTS:
        controller = Controller(OvercookedGridworld.from_layout_name(layout))
        probes = collect_send_probes(controller)
        baseline_rate, baselines = evaluate_baselines(controller, HELD_OUT_SEEDS)

        # Cached per layout: bisection revisits rates across costs.
        @cache
        def messages_at_rate(rate):
            return evaluate(
                controller,
                CALIBRATION_SEEDS,
                eligible_random_policy(rate),
                repeats=RANDOM_REPEATS,
            )["means"]["messages_sent"]

        sweep = []
        for cost in costs:
            policy = fit_send_policy(probes, cost)
            # Calibration seeds set the random gate's budget, so the held-out
            # comparison is not tuned on the seeds it reports.
            calibration = evaluate(controller, CALIBRATION_SEEDS, lambda *_: policy)
            target_messages = calibration["means"]["messages_sent"]
            rate, matched_messages = match_message_budget(messages_at_rate, target_messages)
            learned = evaluate(controller, HELD_OUT_SEEDS, lambda *_: policy)
            random_matched = evaluate(
                controller,
                HELD_OUT_SEEDS,
                eligible_random_policy(rate),
                repeats=RANDOM_REPEATS,
            )
            conditions = {
                **baselines,
                "learned": learned,
                "random_matched_to_learned": random_matched,
            }
            sweep.append(
                {
                    "message_cost": cost,
                    "calibrated_random_rate": rate,
                    "calibration_learned_messages": target_messages,
                    "calibration_random_messages": matched_messages,
                    "learned": learned,
                    "random_matched_to_learned": random_matched,
                    # All conditions share the held-out seeds, so every row
                    # of the sweep compares them on the same episodes.
                    "adjusted_return": {
                        name: summarize(adjusted_returns(condition, cost))
                        for name, condition in conditions.items()
                    },
                    "comparisons": {
                        f"{policy}_vs_{reference}": compare_rewards(
                            conditions[policy], conditions[reference], cost
                        )
                        for policy, reference in SWEEP_COMPARISONS
                    },
                }
            )
        results["layouts"][layout] = {
            "training_probes": sum(len(samples) for samples in probes.values()),
            "conflict_rule_random_rate": baseline_rate,
            "baselines": baselines,
            "cost_sweep": sweep,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("heuristic", "learned"))
    parser.add_argument("--output", type=Path, help="Write JSON to this file.")
    parser.add_argument(
        "--costs", type=float, nargs="+", help="Message costs for the learned sweep."
    )
    args = parser.parse_args()
    if args.costs is not None and (
        args.experiment != "learned"
        or any(not isfinite(cost) or cost < 0 for cost in args.costs)
    ):
        parser.error("--costs requires the learned experiment and finite nonnegative values")
    results = (
        run_heuristic()
        if args.experiment == "heuristic"
        else run_learned(DEFAULT_COSTS if args.costs is None else args.costs)
    )
    output = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
