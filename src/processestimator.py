from collections import defaultdict
from typing import List, Dict, Tuple


class MarkovProcessEstimator:
    """
    Estimates transition probabilities for a Markov Process from observed episodes.
    """

    def __init__(self):
        # Track all unique states observed
        self.states = set()

        # Count transitions: transition_counts[state_from][state_to] = count
        self.transition_counts = defaultdict(lambda: defaultdict(int))

        # Count total times each state was visited (as a source state)
        self.state_counts = defaultdict(int)

        # Store episodes for reference
        self.episodes = []

    def add_episode(self, episode: List[str]) -> None:
        """
        Add an episode (sequence of states) to the estimator.

        Args:
            episode: List of state names, e.g., ['Home', 'Coffee', 'Chat']
        """
        if len(episode) < 2:
            # Need at least 2 states to observe a transition
            return

        self.episodes.append(episode)

        # Process each transition in the episode
        for i in range(len(episode) - 1):
            state_from = episode[i]
            state_to = episode[i + 1]

            # Update state space
            self.states.add(state_from)
            self.states.add(state_to)

            # Update counts
            self.transition_counts[state_from][state_to] += 1
            self.state_counts[state_from] += 1

    def get_transition_probability(self, state_from: str, state_to: str) -> float:
        """
        Get estimated probability of transitioning from state_from to state_to.

        Returns:
            Probability estimate, or 0.0 if state_from was never observed
        """
        if self.state_counts[state_from] == 0:
            return 0.0

        count = self.transition_counts[state_from][state_to]
        total = self.state_counts[state_from]

        return count / total

    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get the full estimated transition probability matrix.

        Returns:
            Nested dictionary: transition_matrix[state_from][state_to] = probability
        """
        matrix = {}

        for state_from in self.states:
            matrix[state_from] = {}
            for state_to in self.states:
                matrix[state_from][state_to] = self.get_transition_probability(
                    state_from, state_to
                )

        return matrix

    def get_state_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics for each state.

        Returns:
            Dictionary with counts and probabilities for each state
        """
        stats = {}

        for state in self.states:
            total_count = self.state_counts[state]
            transitions = {}

            for next_state in self.states:
                count = self.transition_counts[state][next_state]
                prob = self.get_transition_probability(state, next_state)

                if count > 0:  # Only include observed transitions
                    transitions[next_state] = {"count": count, "probability": prob}

            stats[state] = {
                "total_occurrences": total_count,
                "transitions": transitions,
            }

        return stats

    def print_summary(self) -> None:
        """
        Print a human-readable summary of the estimated Markov Process.
        """
        print(f"State Space: {sorted(self.states)}")
        print(f"Total Episodes: {len(self.episodes)}\n")

        print("Transition Probabilities:")
        print("-" * 60)

        for state_from in sorted(self.states):
            if self.state_counts[state_from] == 0:
                continue

            print(
                f"\nFrom '{state_from}' (observed {self.state_counts[state_from]} times):"
            )

            for state_to in sorted(self.states):
                prob = self.get_transition_probability(state_from, state_to)
                count = self.transition_counts[state_from][state_to]

                if count > 0:
                    print(f"  → '{state_to}': {prob:.3f} ({count} times)")


# Example usage with the episodes from the text
if __name__ == "__main__":
    estimator = MarkovProcessEstimator()

    # Add the three episodes from your example
    episodes = [
        [
            "Home",
            "Coffee",
            "Coffee",
            "Chat",
            "Chat",
            "Coffee",
            "Computer",
            "Computer",
            "Home",
        ],
        [
            "Computer",
            "Computer",
            "Chat",
            "Chat",
            "Coffee",
            "Computer",
            "Computer",
            "Computer",
        ],
        ["Home", "Home", "Coffee", "Chat", "Computer", "Coffee", "Coffee"],
    ]

    for episode in episodes:
        estimator.add_episode(episode)

    # Print summary
    estimator.print_summary()

    # Example: Get specific transition probability
    print("\n" + "=" * 60)
    print(
        f"\nP(Home → Coffee) = {estimator.get_transition_probability('Home', 'Coffee'):.3f}"
    )
    print(
        f"P(Coffee → Chat) = {estimator.get_transition_probability('Coffee', 'Chat'):.3f}"
    )
