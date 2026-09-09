"""
Chapter 4 -- Markov Decision Processes
gridworld_mdp.py

Implements the complete "1x4 slippery strip" MDP defined explicitly in
the textbook:

    S = {0, 1, 2, 3}                 (cell 3 is terminal / the goal)
    A = {LEFT, RIGHT}
    P: intended move succeeds with probability 0.9;
       with probability 0.1 the agent stays in place instead.
       Moving off either end of the strip also leaves the agent in place.
    R: -1 for any transition that does not land on state 3, +10 for any
       transition that lands on state 3.
    gamma = 0.95

This is a *model* class: unlike code/envs/gridworld.py (built for an
agent to interact with step-by-step), this class exposes P(s'|s,a) and
R(s,a,s') directly, because Chapter 4 is about the mathematical MDP
object itself, not about an agent's interaction loop yet.
"""

from typing import Dict, Tuple

LEFT, RIGHT = "LEFT", "RIGHT"


class SlipperyStripMDP:
    def __init__(self, n_states: int = 4, slip_prob: float = 0.1,
                 goal_state: int = 3, gamma: float = 0.95):
        self.states = list(range(n_states))
        self.actions = [LEFT, RIGHT]
        self.goal_state = goal_state
        self.slip_prob = slip_prob
        self.success_prob = 1.0 - slip_prob
        self.gamma = gamma

    def is_terminal(self, state: int) -> bool:
        return state == self.goal_state

    def _intended_next_state(self, state: int, action: str) -> int:
        """Where the agent would land if the action succeeds (clamped at edges)."""
        if action == LEFT:
            return max(state - 1, 0)
        elif action == RIGHT:
            return min(state + 1, self.states[-1])
        raise ValueError(f"Unknown action: {action}")

    def transition_prob(self, state: int, action: str) -> Dict[int, float]:
        """
        Returns P(s' | state, action) as a dict {next_state: probability}.
        Terminal states have no outgoing transitions (episode has ended).
        """
        if self.is_terminal(state):
            return {}

        intended = self._intended_next_state(state, action)
        if intended == state:
            # Already at an edge in the direction of travel: "slipping" has
            # nowhere else to go, so the move fails to no observable effect.
            return {state: 1.0}

        return {
            intended: self.success_prob,
            state: self.slip_prob,
        }

    def reward(self, state: int, action: str, next_state: int) -> float:
        """R(s, a, s') exactly as specified in the chapter."""
        return 10.0 if next_state == self.goal_state else -1.0

    def expected_reward(self, state: int, action: str) -> float:
        """E[R | s, a] = sum_{s'} P(s'|s,a) * R(s,a,s')."""
        probs = self.transition_prob(state, action)
        return sum(p * self.reward(state, action, s_next) for s_next, p in probs.items())


if __name__ == "__main__":
    mdp = SlipperyStripMDP()

    # --- Chapter 4 worked example ---
    probs = mdp.transition_prob(state=1, action=RIGHT)
    print("Book worked example: P(s' | s=1, a=RIGHT) =", probs)
    print("  (book says {2: 0.9, 1: 0.1})")

    # --- Chapter 4, Exercise 2 ---
    probs_ex2 = mdp.transition_prob(state=2, action=RIGHT)
    print("\nExercise 2: P(s' | s=2, a=RIGHT) =", probs_ex2)
    print(f"  P(s'=3|s=2,RIGHT) = {probs_ex2.get(3, 0.0):.2f}, "
          f"P(s'=2|s=2,RIGHT) = {probs_ex2.get(2, 0.0):.2f}, "
          f"sum = {sum(probs_ex2.values()):.2f}")

    # --- Chapter 4, Exercise 3 ---
    er = mdp.expected_reward(state=2, action=RIGHT)
    print(f"\nExercise 3: E[R | s=2, a=RIGHT] = {er:.2f}")

    # A quick sanity sweep over every non-terminal state/action pair
    print("\nFull transition + expected-reward table:")
    for s in mdp.states:
        if mdp.is_terminal(s):
            print(f"  state {s}: terminal")
            continue
        for a in mdp.actions:
            p = mdp.transition_prob(s, a)
            er = mdp.expected_reward(s, a)
            print(f"  state {s}, action {a:5s}: P={p}, E[R]={er:.2f}")
