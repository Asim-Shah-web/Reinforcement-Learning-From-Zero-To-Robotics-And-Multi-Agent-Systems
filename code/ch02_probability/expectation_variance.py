"""
Chapter 2 -- Probability for RL
expectation_variance.py

Small, reusable expectation/variance helpers for discrete reward
distributions, plus the worked example and Exercise 5 from the chapter.

    E[X]   = sum_x  x * P(x)
    Var(X) = E[X^2] - (E[X])^2
"""

from typing import Dict


def expectation(distribution: Dict[float, float]) -> float:
    """distribution: {value: probability, ...}. Returns E[X]."""
    _assert_valid_distribution(distribution)
    return sum(value * prob for value, prob in distribution.items())


def variance(distribution: Dict[float, float]) -> float:
    """distribution: {value: probability, ...}. Returns Var(X)."""
    _assert_valid_distribution(distribution)
    mean = expectation(distribution)
    mean_of_square = sum((value ** 2) * prob for value, prob in distribution.items())
    return mean_of_square - mean ** 2


def _assert_valid_distribution(distribution: Dict[float, float]) -> None:
    total = sum(distribution.values())
    assert abs(total - 1.0) < 1e-9, f"Probabilities must sum to 1, got {total}"


if __name__ == "__main__":
    # --- Chapter 2 worked example: expected reward ---
    reward_dist = {10.0: 0.7, -1.0: 0.3}
    print("Worked example (reward distribution {+10: 0.7, -1: 0.3}):")
    print(f"  E[R] = {expectation(reward_dist):.4f}  (book says 6.7)")

    # --- Chapter 2, Exercise 5 ---
    ex5_dist = {5.0: 0.6, -2.0: 0.4}
    print("\nExercise 5 (reward distribution {+5: 0.6, -2: 0.4}):")
    print(f"  E[R]     = {expectation(ex5_dist):.4f}")
    print(f"  Var(R)   = {variance(ex5_dist):.4f}")
