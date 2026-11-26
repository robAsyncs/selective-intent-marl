"""Compare silent and always-send scripted agents in two Overcooked layouts."""

import argparse
import json
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from random import Random
from statistics import mean

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState

LAYOUTS = ("cramped_room", "coordination_ring")
# Ticks per episode. Both agents act once per tick.
HORIZON = 400
# Paired seeds for the silent vs always-send pilot.
SCENARIOS = 24
# Tasks whose target only one agent can usefully claim at a time, so two
# agents choosing the same target duplicate work. "add_onion" is left out:
# both agents can add onions to a pot with room for more than one.
EXCLUSIVE_TASKS = {
    "fetch_onion",
    "fetch_dish",
    "start_soup",
    "plate_soup",
    "finish_filling",
}


@dataclass(frozen=True)
class Intent:
    """A private plan: a task name and the kitchen tile it interacts with."""

    task: str
    target: tuple[int, int] | None

    def conflicts_with(self, other: "Intent | None") -> bool:
        """True when both intents claim the same tile with exclusive tasks."""
        return (
            other is not None
            and self.target == other.target
            and self.task in EXCLUSIVE_TASKS
            and other.task in EXCLUSIVE_TASKS
        )


@dataclass
class Metrics:
    """Per-episode totals reported for every condition."""

    # Sum of both agents' sparse rewards (20 per delivered soup).
    team_return: int = 0
    completed_orders: int = 0
    messages_sent: int = 0
    blocked_moves: int = 0
    duplicate_commitments: int = 0


class Controller:
    """Scripted task choice and shortest-path movement, shared by every condition.

    The controller is fixed across conditions, so any difference in outcome
    comes from whether a message was sent.
    """

    def __init__(self, mdp):
        self.mdp = mdp
        self.floor = frozenset(mdp.get_valid_player_positions())

        # Per-instance cache: every argument is hashable, and routes depend only
        # on the static layout plus these arguments.
        self.route = lru_cache(maxsize=8192)(self._route)

    def _route(self, position, facing, target, occupied, directions):
        """Return the number of steps and the first action needed to interact.

        Breadth-first search over (tile, facing) pairs, because an agent must
        face the target tile to interact with it. The step count includes the
        final INTERACT. ``occupied`` is a tile to route around, such as the
        teammate's position. ``directions`` fixes the expansion order, which
        breaks ties between equally short routes. Returns None if the target
        cannot be reached.
        """
        # Queue entries: (tile, facing, first action taken, steps so far).
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
        # get_pot_states returns a defaultdict, so missing keys are empty lists.
        pots = self.mdp.get_pot_states(state)
        # Pots that can still take an onion. A pot with three onions waits for
        # an INTERACT ("start_soup") before it cooks.
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
        """Choose an intended task and the action to take this tick.

        Picks the nearest reachable candidate task. ``message`` is the
        teammate's announced intent, if one was sent this tick. ``back_off``
        lists directions to try stepping aside after a collision.
        """
        player = state.players[index]
        teammate = state.players[1 - index]
        # After a collision, one player tries to step aside before choosing a task.
        for direction in back_off:
            tile = Action.move_in_direction(player.position, direction)
            if tile in self.floor and tile != teammate.position:
                return None, direction

        options = []
        for intent in self.candidates(state, player):
            # Plan as if the teammate were absent. Collisions are resolved by
            # the simulator and the back-off step above.
            route = self.route(
                player.position, player.orientation, intent.target, None, directions
            )
            if route is None:
                continue
            distance, action = route
            if intent.conflicts_with(message):
                # Give way if the sender can get there as quickly while avoiding
                # us. This is the only place a message changes behavior.
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
        """Return the destination before the simulator resolves player collisions.

        Used to count moves that a collision cancels.
        """
        if action == Action.INTERACT:
            return player.position
        tile = Action.move_in_direction(player.position, action)
        return tile if tile in self.floor else player.position


# The gate sees the shared kitchen state and the sender's own intent. The
# receiver has yet to choose either its private intent or its action.
SendPolicy = Callable[[Controller, OvercookedState, int, Intent | None], bool]


def silent(controller, state, sender, intent):
    """Never send. The no-communication baseline."""
    return False


def always_send(controller, state, sender, intent):
    """Send every tick. The full sequential-communication baseline."""
    return True


def competition_distances(
    controller: Controller,
    state: OvercookedState,
    sender: int,
    intent: Intent | None,
) -> tuple[int, int] | None:
    """Find a reachable exclusive target the teammate could also use.

    Returns (sender steps, teammate steps) when a message could change the
    receiver's choice, or None otherwise. A message only redirects the
    receiver when the receiver would pick the same exclusive target and the
    sender can get there at least as fast while routing around the receiver.
    This mirrors the give-way test in Controller.decide.
    """
    if intent is None or intent.task not in EXCLUSIVE_TASKS:
        return None

    player, teammate = state.players[sender], state.players[1 - sender]
    # Candidate eligibility depends on visible held objects and the kitchen.
    # The teammate's task choice and route tie-breaking are private.
    if not any(
        intent.conflicts_with(task) for task in controller.candidates(state, teammate)
    ):
        return None

    # A fixed direction order suffices because only shortest distances matter.
    directions = tuple(Direction.ALL_DIRECTIONS)
    sender_route = controller.route(
        player.position, player.orientation, intent.target, teammate.position, directions
    )
    teammate_route = controller.route(
        teammate.position, teammate.orientation, intent.target, None, directions
    )
    if (
        sender_route is None
        or teammate_route is None
        or sender_route[0] > teammate_route[0]
    ):
        return None
    return sender_route[0], teammate_route[0]


def conflict_rule(controller, state, sender, intent):
    """Heuristic gate: send only when the message could redirect the receiver."""
    return competition_distances(controller, state, sender, intent) is not None


def run_episode(
    controller,
    seed,
    send_policy: SendPolicy,
    *,
    capture_trace: bool = False,
    observe_reward: Callable[[int], None] | None = None,
):
    """Run an episode with optional reward observation and first-message tracing.

    Each tick, the sender picks its intent, the gate decides whether to send
    it, and then the receiver picks its intent with or without the message.
    Both actions are applied together. ``observe_reward`` receives the team
    reward from every tick, in order.

    Every random draw comes from ``seed`` and is made the same way whatever
    the send policy does. Two conditions run with the same seed therefore
    differ only through the messages.
    """
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
        # After a collision, one randomly chosen player steps aside next tick.
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
        send = send_policy(controller, state, sender, sender_intent)
        message = (sender_intent or Intent("idle", None)) if send else None
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

        if capture_trace and message is not None and trace is None:
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
        reward = sum(info["sparse_reward_by_agent"])
        metrics.team_return += reward
        if observe_reward is not None:
            observe_reward(reward)
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
            silent_metrics, _ = run_episode(controller, seed, silent)
            send_metrics, episode_trace = run_episode(
                controller, seed, always_send, capture_trace=trace is None
            )
            episodes.append(
                {
                    "seed": seed,
                    "silent": asdict(silent_metrics),
                    "always_send": asdict(send_metrics),
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
