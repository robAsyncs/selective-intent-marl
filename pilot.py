"""Compare silent and always-send scripted agents in two Overcooked layouts."""

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from random import Random
from statistics import mean

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

LAYOUTS = ("cramped_room", "coordination_ring")
HORIZON = 400
SCENARIOS = 24
EXCLUSIVE_TASKS = {
    "fetch_onion",
    "fetch_dish",
    "start_soup",
    "plate_soup",
    "finish_filling",
}


@dataclass(frozen=True)
class Intent:
    task: str
    target: tuple[int, int] | None

    def conflicts_with(self, other):
        return (
            other is not None
            and self.target == other.target
            and self.task in EXCLUSIVE_TASKS
            and other.task in EXCLUSIVE_TASKS
        )


@dataclass
class Metrics:
    team_return: int = 0
    completed_orders: int = 0
    messages_sent: int = 0
    blocked_moves: int = 0
    duplicate_commitments: int = 0


class Controller:
    def __init__(self, mdp):
        self.mdp = mdp
        self.floor = frozenset(mdp.get_valid_player_positions())

        self.route = lru_cache(maxsize=8192)(self._route)

    def _route(self, position, facing, target, occupied, directions):
        """Shortest route to an interaction, accounting for facing and floor tiles."""
        queue = deque([(position, facing, None, 0)])
        visited = {(position, facing)}
        while queue:
            tile, orientation, first, distance = queue.popleft()
            if Action.move_in_direction(tile, orientation) == target:
                return distance + 1, Action.INTERACT if first is None else first
            for direction in directions:
                destination = Action.move_in_direction(tile, direction)
                if destination == occupied:
                    continue
                next_tile = destination if destination in self.floor else tile
                node = (next_tile, direction)
                if node not in visited:
                    visited.add(node)
                    queue.append(
                        (*node, direction if first is None else first, distance + 1)
                    )
        return None

    def candidates(self, state, player):
        pots = self.mdp.get_pot_states(state)
        open_pots = pots["empty"] + pots["1_items"] + pots["2_items"]
        if not player.has_object():
            if pots["ready"] or pots["cooking"]:
                yield from (
                    Intent("fetch_dish", tile)
                    for tile in self.mdp.get_dish_dispenser_locations()
                )
            yield from (Intent("start_soup", tile) for tile in pots["3_items"])
            if open_pots:
                yield from (
                    Intent("fetch_onion", tile)
                    for tile in self.mdp.get_onion_dispenser_locations()
                )
        elif player.get_object().name == "onion":
            for tile in open_pots:
                task = "finish_filling" if tile in pots["2_items"] else "add_onion"
                yield Intent(task, tile)
        elif player.get_object().name == "dish":
            yield from (
                Intent("plate_soup", tile) for tile in pots["ready"] + pots["cooking"]
            )
        else:
            yield from (
                Intent("deliver_soup", tile)
                for tile in self.mdp.get_serving_locations()
            )

    def decide(self, state, index, directions, message=None, back_off=()):
        player = state.players[index]
        teammate = state.players[1 - index]
        for direction in back_off:
            tile = Action.move_in_direction(player.position, direction)
            if tile in self.floor and tile != teammate.position:
                return None, direction

        options = []
        for intent in self.candidates(state, player):
            args = player.position, player.orientation, intent.target
            route = self.route(*args, None, directions)
            if route is None:
                continue
            distance, action = route
            if intent.conflicts_with(message):
                sender_route = self.route(
                    teammate.position,
                    teammate.orientation,
                    intent.target,
                    player.position,
                    directions,
                )
                if sender_route is not None and sender_route[0] <= distance:
                    continue
            if (
                intent.task == "plate_soup"
                and action == Action.INTERACT
                and not state.get_object(intent.target).is_ready
            ):
                action = Action.STAY
            options.append((distance, intent, action))

        if options:
            _, intent, action = min(options, key=lambda option: option[0])
            return intent, action
        return None, Action.STAY

    def attempted_position(self, player, action):
        if action == Action.INTERACT:
            return player.position
        tile = Action.move_in_direction(player.position, action)
        return tile if tile in self.floor else player.position


def run_episode(controller, seed, always_send):
    rng = Random(seed)
    directions = tuple(
        tuple(rng.sample(list(Direction.ALL_DIRECTIONS), 4)) for _ in range(2)
    )
    state = controller.mdp.get_standard_start_state()
    for player, order in zip(state.players, directions):
        player.update_pos_and_or(player.position, order[0])

    metrics = Metrics()
    trace = None
    collision = False
    for tick in range(HORIZON):
        sender = (seed + tick) % 2
        receiver = 1 - sender
        # Draw on every tick so paired scenarios share recovery randomness.
        yield_player = rng.randrange(2)
        back_off_directions = rng.sample(list(Direction.ALL_DIRECTIONS), 4)
        back_off_player = yield_player if collision else None
        intent, send_action = controller.decide(
            state,
            sender,
            directions[sender],
            back_off=back_off_directions if sender == back_off_player else (),
        )
        message = (intent or Intent("idle", None)) if always_send else None
        receive_intent, receive_action = controller.decide(
            state,
            receiver,
            directions[receiver],
            message,
            back_off=back_off_directions if receiver == back_off_player else (),
        )
        metrics.messages_sent += message is not None
        metrics.duplicate_commitments += intent is not None and intent.conflicts_with(
            receive_intent
        )

        if message is not None and trace is None:
            silent_intent, silent_action = controller.decide(
                state,
                receiver,
                directions[receiver],
                back_off=back_off_directions if receiver == back_off_player else (),
            )
            if silent_action != receive_action:
                trace = {
                    "seed": seed,
                    "tick": tick,
                    "back_off_player": back_off_player,
                    "back_off_directions": back_off_directions if collision else (),
                    "receiver_directions": directions[receiver],
                    "sender": sender,
                    "receiver": receiver,
                    "state": state.to_dict(),
                    "message": asdict(message),
                    "silent_receiver_intent": asdict(silent_intent)
                    if silent_intent is not None
                    else None,
                    "silent_receiver_action": silent_action,
                    "messaged_receiver_intent": asdict(receive_intent)
                    if receive_intent is not None
                    else None,
                    "messaged_receiver_action": receive_action,
                }

        actions = [None, None]
        actions[sender], actions[receiver] = send_action, receive_action
        old_positions = tuple(player.position for player in state.players)
        attempted = tuple(
            controller.attempted_position(p, a) for p, a in zip(state.players, actions)
        )
        next_state, info = controller.mdp.get_state_transition(state, tuple(actions))
        collision = controller.mdp.is_transition_collision(old_positions, attempted)
        if collision:
            metrics.blocked_moves += sum(
                old != proposed and new.position == old
                for old, proposed, new in zip(
                    old_positions, attempted, next_state.players
                )
            )
        metrics.team_return += sum(info["sparse_reward_by_agent"])
        metrics.completed_orders += sum(info["event_infos"]["soup_delivery"])
        state = next_state
    return metrics, trace


def run_pilot():
    results = {"horizon": HORIZON, "scenario_count": SCENARIOS, "layouts": {}}
    for layout in LAYOUTS:
        controller = Controller(OvercookedGridworld.from_layout_name(layout))
        episodes = []
        trace = None
        for seed in range(SCENARIOS):
            silent, _ = run_episode(controller, seed, always_send=False)
            always_send, episode_trace = run_episode(controller, seed, always_send=True)
            episodes.append(
                {
                    "seed": seed,
                    "silent": asdict(silent),
                    "always_send": asdict(always_send),
                }
            )
            if trace is None:
                trace = episode_trace
        averages = {
            condition: {
                metric: mean(episode[condition][metric] for episode in episodes)
                for metric in asdict(Metrics())
            }
            for condition in ("silent", "always_send")
        }
        results["layouts"][layout] = {
            "means": averages,
            "trace": trace,
            "episodes": episodes,
        }
        controller.route.cache_clear()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Write JSON to this file. Defaults to stdout."
    )
    args = parser.parse_args()
    output = json.dumps(run_pilot(), indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
