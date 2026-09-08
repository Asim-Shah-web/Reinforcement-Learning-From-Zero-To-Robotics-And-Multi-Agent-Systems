"""Run many random-agent episodes and plot their total rewards."""

import matplotlib.pyplot as plt
import numpy as np

from envs.gridworld import GridWorld


def run_episode(env, rng):
    """Run one episode with a uniformly random agent."""
    env.reset()

    total_reward = 0.0
    terminated = False

    while not terminated:
        action = int(rng.choice(GridWorld.ACTIONS))
        _, reward, terminated, _ = env.step(action)
        total_reward += reward

    return total_reward


def main():
    """Run 200 random episodes and save the return plot."""
    env = GridWorld()
    rng = np.random.default_rng(0)

    rewards = []

    # Preview only: proper "return" with discounting is not defined until Chapter 5, so this just sums raw rewards per episode.
    for _ in range(200):
        rewards.append(run_episode(env, rng))

    plt.figure()
    plt.plot(range(1, 201), rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Random Agent: Total Reward per Episode")
    plt.tight_layout()
    plt.savefig("code/ch01_intro/random_agent_returns.png")
    plt.close()


if __name__ == "__main__":
    main()