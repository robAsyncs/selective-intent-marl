"""Evaluate selective send policies on paired Overcooked scenarios."""

import argparse
import json
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from random import Random
from statistics import mean

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from learned import (
    DEFAULT_COSTS,
    PRIOR_WEIGHT,
    PROBES_PER_EPISODE,
    ROLLOUT_RATES,
    TRAINING_SEEDS,
    fit_send_values,
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

CALIBRATION_SEEDS = range(100, 124)
RANDOM_REPEATS = 8
HELD_OUT_SEEDS = range(2000, 2048)


def random_policy(rate, seed, repeat):
    # Draw independently of the simulator RNG so each condition shares setup
    # and collision recovery draws with its paired scenarios.
    rng = Random(f"random-send:{seed}:{repeat}")

    def send(controller, state, sender, intent):
        return rng.random() < rate

    return send


def evaluate(controller, seeds, make_policy, repeats=1):
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
        "episodes": episodes,
    }


def compare_rewards(policy, reference, message_cost=0):
    """Bootstrap communication-adjusted reward differences paired by seed."""
    policy_episodes = policy["episodes"]
    reference_episodes = reference["episodes"]
    policy_by_seed = {episode["seed"]: episode for episode in policy_episodes}
    reference_by_seed = {episode["seed"]: episode for episode in reference_episodes}
    if len(policy_by_seed) != len(policy_episodes) or len(reference_by_seed) != len(
        reference_episodes
    ):
        raise ValueError("Reward comparisons require one episode per seed per condition")
    if not policy_by_seed or policy_by_seed.keys() != reference_by_seed.keys():
        raise ValueError("Reward comparisons require the same nonempty set of seeds")

    deltas = []
    for seed in sorted(policy_by_seed):
        trial, baseline = policy_by_seed[seed], reference_by_seed[seed]
        reward_delta = trial["team_return"] - baseline["team_return"]
        message_delta = trial["messages_sent"] - baseline["messages_sent"]
        deltas.append(reward_delta - message_cost * message_delta)
    rng = Random(0)
    resamples = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
    return {
        "mean_difference": mean(deltas),
        "bootstrap_95_percent_interval": [resamples[50], resamples[1949]],
    }


def evaluate_baselines(controller, seeds, calibration_seeds=CALIBRATION_SEEDS):
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


def adjusted_return(condition, cost):
    means = condition["means"]
    return means["team_return"] - cost * means["messages_sent"]


def run_heuristic():
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
            "rule_vs_random_reward": compare_rewards(rule, conditions["random_matched"]),
        }
    return results


def run_learned(costs=DEFAULT_COSTS):
    results = {
        "horizon": HORIZON,
        "training_seeds": list(TRAINING_SEEDS),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "evaluation_seeds": list(HELD_OUT_SEEDS),
        "probes_per_training_episode": PROBES_PER_EPISODE,
        "training_rollout_rates": ROLLOUT_RATES,
        "prior_weight": PRIOR_WEIGHT,
        "random_repeats_per_seed": RANDOM_REPEATS,
        "layouts": {},
    }
    for layout in LAYOUTS:
        controller = Controller(OvercookedGridworld.from_layout_name(layout))
        values = fit_send_values(controller)
        baseline_rate, baselines = evaluate_baselines(controller, HELD_OUT_SEEDS)
        sweep = []
        for cost in costs:
            policy = values.policy(cost)
            calibration = evaluate(controller, CALIBRATION_SEEDS, lambda *_: policy)
            rate = calibration["means"]["messages_sent"] / HORIZON
            learned = evaluate(controller, HELD_OUT_SEEDS, lambda *_: policy)
            random_matched = evaluate(
                controller,
                HELD_OUT_SEEDS,
                lambda seed, repeat: random_policy(rate, seed, repeat),
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
                    "learned": learned,
                    "random_matched_to_learned": random_matched,
                    "mean_adjusted_return": {
                        name: adjusted_return(condition, cost)
                        for name, condition in conditions.items()
                    },
                    "learned_vs_random_adjusted_return": compare_rewards(
                        learned, random_matched, cost
                    ),
                }
            )
        results["layouts"][layout] = {
            "training_probes": sum(totals.samples for totals in values.by_task.values()),
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
