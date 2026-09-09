"""
Chapter 3 -- The Markov Property and Markov Chains
weather_markov_chain.py

Implements the Sunny/Rainy Markov chain from the textbook:

    P(Sunny | Sunny) = 0.8      P(Rainy | Sunny) = 0.2
    P(Sunny | Rainy) = 0.4      P(Rainy | Rainy) = 0.6

Demonstrates:
  - computing the probability of a specific multi-day sequence using the
    Markov property (each step conditions only on the immediately
    preceding day, per Chapter 3),
  - computing an n-step transition probability by summing over every
    intermediate path, exactly as done by hand in the chapter,
  - the same n-step result via transition-matrix power (numpy), so you
    can see the "sum over paths" trick and the "matrix multiply" trick
    are the same computation.
"""

import numpy as np

STATES = ["Sunny", "Rainy"]
IDX = {s: i for i, s in enumerate(STATES)}

# Row i = "given state i today", columns = probability of each state tomorrow.
TRANSITION_MATRIX = np.array([
    [0.8, 0.2],   # from Sunny
    [0.4, 0.6],   # from Rainy
])


def transition_prob(from_state: str, to_state: str) -> float:
    return TRANSITION_MATRIX[IDX[from_state], IDX[to_state]]


def sequence_probability(sequence, start_prob: float = 1.0) -> float:
    """
    P(sequence[0], sequence[1], ..., sequence[-1]) using the Markov
    property: each step's probability depends only on the previous state.
    `start_prob` is P(sequence[0]) -- 1.0 if we're told the chain starts
    there for certain, as in the book's examples.
    """
    prob = start_prob
    for t in range(len(sequence) - 1):
        prob *= transition_prob(sequence[t], sequence[t + 1])
    return prob


def n_step_transition_by_summing_paths(start_state: str, end_state: str, n: int) -> float:
    """
    Compute P(state_n = end_state | state_0 = start_state) by explicitly
    summing over every possible path of length n (the "by hand" method
    used in the chapter for n=2). Only practical for small n / few states,
    which is exactly why the matrix-power method below exists.
    """
    if n == 0:
        return 1.0 if start_state == end_state else 0.0
    if n == 1:
        return transition_prob(start_state, end_state)

    total = 0.0
    for intermediate in STATES:
        total += (
            transition_prob(start_state, intermediate)
            * n_step_transition_by_summing_paths(intermediate, end_state, n - 1)
        )
    return total


def n_step_transition_by_matrix_power(start_state: str, end_state: str, n: int) -> float:
    """Same quantity as above, computed via T^n instead of manual recursion."""
    T_n = np.linalg.matrix_power(TRANSITION_MATRIX, n)
    return T_n[IDX[start_state], IDX[end_state]]


if __name__ == "__main__":
    print("Each row of the transition matrix sums to 1:")
    print(TRANSITION_MATRIX, "\nRow sums:", TRANSITION_MATRIX.sum(axis=1))

    # --- Chapter 3 worked example ---
    seq = ["Sunny", "Sunny", "Rainy"]
    print(f"\nP({seq}) = {sequence_probability(seq):.4f}  (book says 0.16)")

    # --- Chapter 3 worked example: 2-step transition by summing paths ---
    p2 = n_step_transition_by_summing_paths("Sunny", "Rainy", 2)
    print(f"\nP(state_2=Rainy | state_0=Sunny), by summing paths = {p2:.4f}  (book says 0.28)")
    p2_matrix = n_step_transition_by_matrix_power("Sunny", "Rainy", 2)
    print(f"Same quantity via matrix power T^2                  = {p2_matrix:.4f}")

    # --- Chapter 3, Exercise 3 ---
    seq_ex3 = ["Rainy", "Sunny", "Sunny"]
    print(f"\nExercise 3: P({seq_ex3}) = {sequence_probability(seq_ex3):.4f}")

    # --- Chapter 3, Exercise 4 ---
    p_ex4 = n_step_transition_by_summing_paths("Rainy", "Sunny", 2)
    p_ex4_matrix = n_step_transition_by_matrix_power("Rainy", "Sunny", 2)
    print(f"Exercise 4: P(state_2=Sunny | state_0=Rainy) = {p_ex4:.4f} "
          f"(matrix check: {p_ex4_matrix:.4f})")
