"""
Chapter 2 -- Probability for RL
trajectory_chain_rule.py

Reproduces the trajectory-probability tree diagram from the textbook:

        s0
       /    \\
   a0=L      a0=R      (P(a0=L)=0.4, P(a0=R)=0.6)
   /  \\      /  \\
  A    B    C    D     (branch probabilities as in the book)

and demonstrates the chain rule:
    P(s0, a0, s1) = P(s0) * P(a0 | s0) * P(s1 | s0, a0)

Run directly to print every leaf-path probability (and confirm they sum
to 1), plus the two exercise answers from Chapter 2.
"""

from itertools import product

# ---- The tree, encoded exactly as in the book's diagram ----

P_S0 = 1.0  # we always start in s0 (P(s0) = 1.0 by construction here)

P_A0 = {
    "L": 0.4,
    "R": 0.6,
}

# P(s1 | s0, a0) for each action branch
P_S1_GIVEN_A0 = {
    "L": {"A": 0.8, "B": 0.2},
    "R": {"C": 0.9, "D": 0.1},
}


def path_probability(a0: str, s1: str) -> float:
    """Chain rule: P(s0, a0, s1) = P(s0) * P(a0 | s0) * P(s1 | s0, a0)."""
    return P_S0 * P_A0[a0] * P_S1_GIVEN_A0[a0][s1]


def all_leaf_probabilities() -> dict:
    """Every complete path through the tree, with its probability."""
    probs = {}
    for a0, next_states in P_S1_GIVEN_A0.items():
        for s1 in next_states:
            probs[(a0, s1)] = path_probability(a0, s1)
    return probs


if __name__ == "__main__":
    print("All leaf-path probabilities (chain rule applied along each branch):\n")
    probs = all_leaf_probabilities()
    for (a0, s1), p in probs.items():
        print(f"  P(s0, a0={a0}, s1={s1}) = {p:.4f}")

    total = sum(probs.values())
    print(f"\nSum over all leaf paths = {total:.4f} (should be 1.0)")

    # --- Chapter 2, worked example in the book text ---
    p_L_A = path_probability("L", "A")
    print(f"\nBook worked example: P(s0, a0=L, s1=A) = {p_L_A:.4f} (book says 0.32)")

    # --- Chapter 2, Exercise 2: P(s0, a0=R, s1=D) ---
    p_R_D = path_probability("R", "D")
    print(f"Exercise 2 answer:   P(s0, a0=R, s1=D) = {p_R_D:.4f}")
