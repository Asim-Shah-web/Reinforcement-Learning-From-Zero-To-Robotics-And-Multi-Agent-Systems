"""Show what happens when the environment provides no reward signal."""

import numpy as np

from envs.gridworld import GridWorld


class ZeroRewardGridWorld(GridWorld):
    """GridWorld variant where every reward is forced to zero."""

    def step(self, action):
        """Take an action while replacing the reward with zero."""
        next_state, _, terminated, info = super().step(action)
        return next_state, 0.0, terminated, info


def run_episode(env, rng):
    """Run one uniformly random episode and return its total reward."""
    env.reset()

    total_reward = 0.0
    terminated = False

    while not terminated:
        action = int(rng.choice(GridWorld.ACTIONS))
        _, reward, terminated, _ = env.step(action)
        total_reward += reward

    return total_reward


def main():
    """Run 50 episodes and verify that every total reward is zero."""
    env = ZeroRewardGridWorld()
    rng = np.random.default_rng(0)

    total_rewards = []

    for _ in range(50):
        total_rewards.append(run_episode(env, rng))

    assert all(reward == 0.0 for reward in total_rewards)

    print("Total reward per episode:")
    print(total_rewards)

    print(
        "\nAll 50 episodes have total reward 0. "
        "With no reward signal, there is nothing for an RL algorithm "
        "to optimize toward."
    )


if __name__ == "__main__":
    main()