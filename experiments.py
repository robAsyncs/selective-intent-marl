"""Evaluate selective send policies on paired Overcooked scenarios."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from random import Random
from statistics import mean

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

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
    """Bootstrap paired scenario differences in communication-adjusted reward."""
    deltas = [
        (trial["team_return"] - baseline["team_return"])
        - message_cost * (trial["messages_sent"] - baseline["messages_sent"])
        for trial, baseline in zip(policy["episodes"], reference["episodes"], strict=True)
    ]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("heuristic",))
    parser.add_argument("--output", type=Path, help="Write JSON to this file.")
    args = parser.parse_args()
    output = json.dumps(run_heuristic(), indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
