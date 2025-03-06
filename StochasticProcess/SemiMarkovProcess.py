# SemiMarkovProcess.py
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from collections import defaultdict

from MarkovChain import MarkovChain

class SemiMarkovProcess:
    """
    Semi-Markov Process with finite state space

    Features:
    - General holding time distributions for each state
    - Embedded Markov chain transition probabilities
    - Trajectory generation with state-time tracking

    :param states: List of state labels
    :param transition_probs: Embedded Markov chain transition probabilities (Dict[from_state, Dict[to_state, prob]])
    :param holding_time_dists: Holding time distributions for each state (Dict[state, sampling_function])
    """
    def __init__(self,
                 states: List[str],
                 transition_probs: Dict[str, Dict[str, float]],
                 holding_time_dists: Dict[str, Callable[[], float]]):

        # Validate inputs
        self._validate_transition_probs(states, transition_probs)
        self._validate_holding_dists(states, holding_time_dists)

        self.states = states
        self.state_indices = {s: i for i, s in enumerate(states)}
        self.transition_probs = transition_probs
        self.holding_time_dists = holding_time_dists
        self.current_state = None

    def _validate_transition_probs(self,
                                  states: List[str],
                                  transition_probs: Dict[str, Dict[str, float]]):
        """Validate transition probability matrix"""
        for s in states:
            if s not in transition_probs:
                raise ValueError(f"Missing transition probabilities for state {s}")

            probs = transition_probs[s].values()
            if not np.isclose(sum(probs), 1.0, atol=1e-6):
                raise ValueError(f"Transition probabilities for state {s} must sum to 1")

    def _validate_holding_dists(self,
                               states: List[str],
                               holding_time_dists: Dict[str, Callable[[], float]]):
        """Validate holding time distributions"""
        for s in states:
            if s not in holding_time_dists:
                raise ValueError(f"Missing holding time distribution for state {s}")

            try:
                sample = holding_time_dists[s]()
                if sample <= 0:
                    raise ValueError(f"Holding time samples must be positive")
            except:
                raise ValueError(f"Invalid holding time distribution for state {s}")

    def generate_trajectory(self,
                           duration: float,
                           initial_state: str) -> List[Tuple[float, str]]:
        """
        Generate state trajectory with holding times

        :param duration: Total simulation time
        :param initial_state: Starting state
        :return: List of (transition_time, state) tuples
        """
        if initial_state not in self.state_indices:
            raise ValueError(f"Invalid initial state: {initial_state}")

        trajectory = [(0.0, initial_state)]
        current_time = 0.0
        current_state = initial_state

        while current_time < duration:
            # Sample holding time
            holding_time = self.holding_time_dists[current_state]()
            transition_time = current_time + holding_time

            if transition_time > duration:
                break

            # Sample next state
            next_probs = self.transition_probs[current_state]
            next_state = np.random.choice(
                list(next_probs.keys()),
                p=list(next_probs.values())
            )

            trajectory.append((transition_time, next_state))
            current_time = transition_time
            current_state = next_state

        return trajectory

    def embedded_chain(self) -> 'MarkovChain':
        """
        Get embedded discrete-time Markov chain

        :return: MarkovChain object representing state transitions
        """
        n = len(self.states)
        transition_matrix = np.zeros((n, n))

        for i, s_from in enumerate(self.states):
            for s_to, prob in self.transition_probs[s_from].items():
                j = self.state_indices[s_to]
                transition_matrix[i, j] = prob

        return MarkovChain(transition_matrix, self.states)

    def time_average_distribution(self,
                                 trajectory: List[Tuple[float, str]]) -> Dict[str, float]:
        """
        Compute empirical time-averaged distribution from trajectory

        :param trajectory: List of (time, state) tuples
        :return: Time-averaged state distribution
        """
        dwell_times = defaultdict(float)
        if not trajectory:
            return {}

        prev_time, current_state = trajectory[0]
        for time, next_state in trajectory[1:]:
            dwell_times[current_state] += time - prev_time
            prev_time, current_state = time, next_state

        total_time = sum(dwell_times.values())
        return {s: t / total_time for s, t in dwell_times.items()}

    def plot_trajectory(self,
                       trajectory: List[Tuple[float, str]],
                       figsize: tuple = (10, 4),
                       save_path: Optional[str] = None) -> None:
        """
        Visualize state trajectory using step plot

        :param trajectory: List of (time, state) tuples
        :param figsize: Figure dimensions
        :param save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for plotting")

        times, states = zip(*trajectory)
        state_labels = sorted(set(states), key=self.states.index)
        numeric_states = [state_labels.index(s) for s in states]

        plt.figure(figsize=figsize)
        plt.step(times, numeric_states, where='post')
        plt.yticks(range(len(state_labels)), state_labels)
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.title("Semi-Markov Process Trajectory")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()