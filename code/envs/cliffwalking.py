"""A standard Cliff Walking environment for reinforcement learning."""

from __future__ import annotations

import numpy as np


class CliffWalking:
    """A 4x12 Cliff Walking environment.

    The agent starts at the bottom-left corner and must reach the
    bottom-right goal. Stepping into the cliff gives a reward of -100
    and sends the agent back to the start without ending the episode.
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTIONS = (UP, DOWN, LEFT, RIGHT)

    def __init__(self, max_steps: int = 500) -> None:
        """Create the Cliff Walking environment.

        Args:
            max_steps: Maximum number of steps in an episode.
        """
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.max_steps = max_steps

        self.cliff = {
            (3, col)
            for col in range(1, self.cols - 1)
        }

        self.agent_position = self.start
        self.steps = 0

    def reset(self) -> tuple[int, int]:
        """Reset the agent to the starting position.

        Returns:
            The initial state.
        """
        self.agent_position = self.start
        self.steps = 0
        return self.agent_position

    def step(self, action: int) -> tuple[tuple[int, int], float, bool, dict]:
        """Take one action in the environment.

        Args:
            action: One of ``UP``, ``DOWN``, ``LEFT``, or ``RIGHT``.

        Returns:
            A tuple containing ``(next_state, reward, terminated, info)``.
        """
        if action not in self.ACTIONS:
            raise ValueError(f"invalid action: {action}")

        row, col = self.agent_position

        if action == self.UP:
            candidate = (row - 1, col)
        elif action == self.DOWN:
            candidate = (row + 1, col)
        elif action == self.LEFT:
            candidate = (row, col - 1)
        else:
            candidate = (row, col + 1)

        if not self._in_bounds(candidate):
            candidate = self.agent_position

        self.steps += 1

        if candidate in self.cliff:
            self.agent_position = self.start
            reward = -100.0
            terminated = False
        else:
            self.agent_position = candidate
            reward = -1.0
            terminated = self.agent_position == self.goal

        if self.steps >= self.max_steps:
            terminated = True

        return self.agent_position, reward, terminated, {}

    def render(self) -> None:
        """Print the current environment as an ASCII grid."""
        for row in range(self.rows):
            cells = []

            for col in range(self.cols):
                position = (row, col)

                if position == self.agent_position:
                    cells.append("A")
                elif position == self.goal:
                    cells.append("G")
                elif position in self.cliff:
                    cells.append("C")
                else:
                    cells.append(".")

            print(" ".join(cells))

        print()

    def _in_bounds(self, position: tuple[int, int]) -> bool:
        """Return whether a position lies inside the grid."""
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols


if __name__ == "__main__":
    """Run a small random-action episode as a self-check."""

    env = CliffWalking()
    state = env.reset()
    trajectory = [state]

    rng = np.random.default_rng(0)
    terminated = False

    while not terminated:
        action = int(rng.choice(CliffWalking.ACTIONS))
        state, reward, terminated, _ = env.step(action)
        trajectory.append(state)

        print(f"Action: {action}, State: {state}, Reward: {reward}")

    print("Trajectory:")
    print(trajectory)