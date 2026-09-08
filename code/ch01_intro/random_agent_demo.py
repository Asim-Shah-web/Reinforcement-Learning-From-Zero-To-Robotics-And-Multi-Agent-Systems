"""Make the agent-environment loop tangible with a random agent.

No learning happens in this script. The agent simply chooses a uniformly
random action at every step.
"""

import numpy as np

from envs.gridworld import GridWorld


def main():
    """Run one episode with a uniformly random agent."""
    env = GridWorld()
    rng = np.random.default_rng()

    state = env.reset()
    trajectory = []
    terminated = False

    print("Step | State | Action | Next State | Reward")
    print("-" * 48)

    step = 0

    while not terminated:
        action = int(rng.choice(GridWorld.ACTIONS))
        next_state, reward, terminated, _ = env.step(action)

        trajectory.append((state, action, reward))

        print(
            f"{step:4d} | {state!s:5} | {action:6d} | "
            f"{next_state!s:10} | {reward:6.1f}"
        )

        state = next_state
        step += 1

    print("\nFull trajectory:")
    print(trajectory)


if __name__ == "__main__":
    main()