"""Sanity tests for the shared textbook environments."""

import numpy as np

from gridworld import GridWorld
from cliffwalking import CliffWalking


def test_gridworld_reset_returns_valid_state():
    """GridWorld reset should return a valid starting state."""
    env = GridWorld(size=4)

    state = env.reset()

    assert state == (0, 0)
    assert 0 <= state[0] < env.size
    assert 0 <= state[1] < env.size


def test_gridworld_wall_keeps_agent_in_place():
    """Moving into the top wall should keep the agent in place."""
    env = GridWorld(size=4)

    state = env.reset()
    next_state, reward, terminated, _ = env.step(GridWorld.UP)

    assert next_state == state
    assert reward == -1.0
    assert not terminated


def test_gridworld_reaches_goal():
    """A scripted path to the goal should end the episode."""
    env = GridWorld(size=2)

    env.reset()

    env.step(GridWorld.DOWN)
    state, reward, terminated, _ = env.step(GridWorld.RIGHT)

    assert state == (1, 1)
    assert reward == 10.0
    assert terminated


def test_cliffwalking_reset_returns_valid_state():
    """Cliff Walking reset should return the starting state."""
    env = CliffWalking()

    state = env.reset()

    assert state == env.start


def test_cliffwalking_cliff_reward():
    """Stepping from the start into the cliff should give -100."""
    env = CliffWalking()

    env.reset()

    next_state, reward, terminated, _ = env.step(CliffWalking.RIGHT)

    assert reward == -100.0
    assert next_state == env.start
    assert not terminated


def test_cliffwalking_goal():
    """A safe path around the cliff should reach the goal."""
    env = CliffWalking(max_steps=100)

    env.reset()

    # Move up to the row above the cliff.
    for _ in range(3):
        state, reward, terminated, _ = env.step(CliffWalking.UP)

    # Move across the grid.
    for _ in range(11):
        state, reward, terminated, _ = env.step(CliffWalking.RIGHT)

    # Move down onto the goal.
    state, reward, terminated, _ = env.step(CliffWalking.DOWN)

    assert state == env.goal
    assert reward == -1.0
    assert terminated


def test_gridworld_random_policy_terminates():
    """A random GridWorld policy should terminate within max_steps."""
    env = GridWorld(size=4, max_steps=50)
    rng = np.random.default_rng(0)

    state = env.reset()
    terminated = False

    for _ in range(env.max_steps):
        action = int(rng.choice(GridWorld.ACTIONS))
        state, reward, terminated, _ = env.step(action)

        if terminated:
            break

    assert terminated


def test_cliffwalking_random_policy_terminates():
    """A random Cliff Walking policy should terminate within max_steps."""
    env = CliffWalking(max_steps=100)
    rng = np.random.default_rng(0)

    state = env.reset()
    terminated = False

    for _ in range(env.max_steps):
        action = int(rng.choice(CliffWalking.ACTIONS))
        state, reward, terminated, _ = env.step(action)

        if terminated:
            break

    assert terminated