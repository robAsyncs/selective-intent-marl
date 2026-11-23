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
# Both agents can add onions to a pot with room for more than one.
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

    def conflicts_with(self, other: "Intent | None") -> bool:
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
        """Return the number of steps and the first action needed to interact."""
        queue = deque([(position, facing, None, 0)])
        visited = {(position, facing)}
        while queue:
            tile, orientation, first_action, distance = queue.popleft()
            if Action.move_in_direction(tile, orientation) == target:
                return distance + 1, (
                    Action.INTERACT if first_action is None else first_action
                )
            for direction in directions:
                destination = Action.move_in_direction(tile, direction)
                if destination == occupied:
                    continue
                # Stepping toward a counter changes facing without moving.
                next_tile = destination if destination in self.floor else tile
                next_state = (next_tile, direction)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append(
                        (
                            next_tile,
                            direction,
                            direction if first_action is None else first_action,
                            distance + 1,
                        )
                    )
        return None

    def candidates(self, state, player):
        """List tasks based on what the player is carrying and which pots are open."""
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
        """Choose an intended task and the action to take this tick."""
        player = state.players[index]
        teammate = state.players[1 - index]
        # After a collision, one player tries to step aside before choosing a task.
        for direction in back_off:
            tile = Action.move_in_direction(player.position, direction)
            if tile in self.floor and tile != teammate.position:
                return None, direction

        options = []
        for intent in self.candidates(state, player):
            route = self.route(
                player.position, player.orientation, intent.target, None, directions
            )
            if route is None:
                continue
            distance, action = route
            if intent.conflicts_with(message):
                # Give way if the sender can get there as quickly while avoiding us.
                sender_route = self.route(
                    teammate.position,
                    teammate.orientation,
                    intent.target,
                    player.position,
                    directions,
                )
                if sender_route is not None and sender_route[0] <= distance:
                    continue
            # Walk to a cooking pot with a dish, then wait until the soup is ready.
            if (
                intent.task == "plate_soup"
                and action == Action.INTERACT
                and not state.get_object(intent.target).is_ready
            ):
                action = Action.STAY
            options.append((distance, intent, action))

        if options:
            # Equal distances keep the order from candidates().
            _, intent, action = min(options, key=lambda option: option[0])
            return intent, action
        return None, Action.STAY

    def attempted_position(self, player, action):
        """Return the destination before the simulator resolves player collisions."""
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
        # Alternate who chooses first. Both actions still happen together.
        sender = (seed + tick) % 2
        receiver = 1 - sender
        # Draw on every tick so paired scenarios share recovery randomness.
        recovery_player = rng.randrange(2)
        back_off_directions = rng.sample(list(Direction.ALL_DIRECTIONS), 4)
        back_off_player = recovery_player if collision else None
        receiver_back_off = back_off_directions if receiver == back_off_player else ()
        sender_intent, sender_action = controller.decide(
            state,
            sender,
            directions[sender],
            back_off=back_off_directions if sender == back_off_player else (),
        )
        message = (sender_intent or Intent("idle", None)) if always_send else None
        receiver_intent, receiver_action = controller.decide(
            state,
            receiver,
            directions[receiver],
            message=message,
            back_off=receiver_back_off,
        )
        metrics.messages_sent += message is not None
        metrics.duplicate_commitments += (
            sender_intent is not None and sender_intent.conflicts_with(receiver_intent)
        )

        # Capture the first action that changes when the same receiver sees a message.
        if message is not None and trace is None:
            silent_intent, silent_action = controller.decide(
                state,
                receiver,
                directions[receiver],
                back_off=receiver_back_off,
            )
            if silent_action != receiver_action:
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
                    "messaged_receiver_intent": asdict(receiver_intent)
                    if receiver_intent is not None
                    else None,
                    "messaged_receiver_action": receiver_action,
                }

        actions = [None, None]
        actions[sender], actions[receiver] = sender_action, receiver_action
        old_positions = tuple(player.position for player in state.players)
        attempted = tuple(
            controller.attempted_position(player, action)
            for player, action in zip(state.players, actions)
        )
        next_state, info = controller.mdp.get_state_transition(state, tuple(actions))
        collision = controller.mdp.is_transition_collision(old_positions, attempted)
        if collision:
            # Count floor moves canceled by the collision.
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
            # Each seed pairs the two conditions on the same starting setup.
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
