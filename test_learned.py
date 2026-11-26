import unittest
from types import SimpleNamespace
from unittest.mock import patch

from experiments import eligible_random_policy, match_message_budget
from learned import Outcome, Probe, Rollout, fit_send_policy, time_bin
from pilot import HORIZON, Intent


class SendPolicyTests(unittest.TestCase):
    @patch("learned.competition_distances", return_value=(1, 1))
    def test_bellman_target_uses_greedy_successor_and_remaining_time(self, _):
        intent = Intent("fetch_dish", (0, 0))
        start = (intent.task, 1, 0)
        successor = ("plate_soup", 1, 0)
        no_reward = Outcome(3, None, ())
        probes = {
            (time_bin(3), start): [
                Probe(3, no_reward, Outcome(1, successor, ()))
            ],
            (time_bin(2), successor): [
                Probe(2, no_reward, Outcome(2, None, ((1, 2),)))
            ],
        }
        early = SimpleNamespace(timestep=HORIZON - 3)
        penultimate = SimpleNamespace(timestep=HORIZON - 2)
        final = SimpleNamespace(timestep=HORIZON - 1)

        low_cost = fit_send_policy(probes, 0.5)
        self.assertTrue(low_cost(None, early, 0, intent))
        self.assertFalse(low_cost(None, penultimate, 0, intent))
        self.assertFalse(low_cost(None, final, 0, intent))
        self.assertFalse(fit_send_policy(probes, 1.0)(None, early, 0, intent))

    def test_terminal_reward_includes_last_tick_and_censors_later_rewards(self):
        terminal = Outcome(3, None, ((1, 4), (2, 7)))
        probe = Probe(3, terminal, terminal)
        self.assertEqual(terminal.target(None, 2), 4)
        self.assertEqual(terminal.target(None, 3), 11)
        self.assertFalse(probe.covers(4))

    def test_replay_stops_before_the_next_eligible_action(self):
        start = ("fetch_dish", 1, 0)
        successor = ("plate_soup", 1, 0)
        rollout = Rollout({1: (start, True), 3: (successor, False)}, [8, 2, 3, 5, 7])
        self.assertEqual(rollout.outcome(1), Outcome(2, successor, ((0, 2), (1, 3))))
        self.assertEqual(rollout.outcome(3), Outcome(2, None, ((0, 5), (1, 7))))

    @patch("learned.competition_distances", side_effect=[(1, 1), None])
    def test_random_baseline_uses_the_same_eligibility(self, _):
        send = eligible_random_policy(1.0)(10, 0)
        intent = Intent("fetch_dish", (0, 0))
        before_end = SimpleNamespace(timestep=HORIZON - 2)
        final = SimpleNamespace(timestep=HORIZON - 1)
        self.assertTrue(send(None, before_end, 0, intent))
        self.assertFalse(send(None, before_end, 0, intent))
        self.assertFalse(send(None, final, 0, intent))

    def test_random_baseline_matches_calibration_message_budget(self):
        self.assertEqual(match_message_budget(lambda rate: 40 * rate, 15), (0.375, 15))
        self.assertEqual(match_message_budget(lambda rate: 40 * rate, 40), (1.0, 40))
        self.assertEqual(match_message_budget(lambda rate: 40 * rate, 0), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
