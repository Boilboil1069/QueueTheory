# MarkovProcess.py
import numpy as np
from typing import Dict, List, Callable, Optional
from scipy.linalg import expm

from .MarkovChain import MarkovChain


class ContinuousTimeMarkovProcess:
    """
    Continuous-time Markov process with finite state space

    Features:
    - State transitions governed by rate matrix Q
    - Embedded Markov chain analysis
    - Stationary distribution calculation

    :param rate_matrix: State transition rate matrix (n x n)
    :param states: List of state labels (length n)
    """

    def __init__(self, rate_matrix: np.ndarray, states: List[str]):
        # Validate rate matrix
        if not np.allclose(rate_matrix.sum(axis=1), 0, atol=1e-6):
            raise ValueError("Rows of rate matrix must sum to zero")
        if (rate_matrix.diagonal() >= 0).any():
            raise ValueError("Diagonal elements must be negative")
        if (rate_matrix - np.diag(rate_matrix.diagonal()) < 0).any():
            raise ValueError("Off-diagonal elements must be non-negative")

        self.Q = rate_matrix.astype(np.float64)
        self.states = states
        self.state_indices = {s: i for i, s in enumerate(states)}
        self.current_state = None

    def _validate_initial_state(self, state: str):
        """Validate and set initial state"""
        if state not in self.state_indices:
            raise ValueError(f"Invalid initial state: {state}")
        self.current_state = self.state_indices[state]

    def generate_trajectory(self, duration: float, initial_state: str) -> List[tuple]:
        """
        Generate state transition trajectory

        :param plot:
        :param duration: Total simulation time
        :param initial_state: Starting state
        :return: List of (transition_time, state) tuples
        """
        self._validate_initial_state(initial_state)

        trajectory = [(0.0, initial_state)]
        current_time = 0.0
        current_idx = self.state_indices[initial_state]

        while current_time < duration:
            # Get transition rates from current state
            rates = self.Q[current_idx].copy()
            rates[current_idx] = 0  # Exclude self-transition
            total_rate = -self.Q[current_idx, current_idx]

            if total_rate == 0:
                break  # Absorbing state

            # Generate next transition time
            dwell_time = np.random.exponential(1/total_rate)
            current_time += dwell_time

            if current_time > duration:
                break

            # Select next state
            transition_probs = rates / total_rate
            next_idx = np.random.choice(len(self.states), p=transition_probs)

            trajectory.append((current_time, self.states[next_idx]))
            current_idx = next_idx

        return trajectory

    def embedded_chain(self) -> MarkovChain:
        """Get embedded discrete-time Markov chain"""
        transition_matrix = self.Q.copy()
        np.fill_diagonal(transition_matrix, 0)
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        return MarkovChain(transition_matrix, self.states)

    def transition_probability(self, t: float) -> np.ndarray:
        """Compute transition probability matrix P(t) = exp(Qt)"""
        return expm(self.Q * t)

    def stationary_distribution(self) -> Dict[str, float]:
        """Compute stationary distribution by solving πQ = 0"""
        n = len(self.states)
        A = np.vstack([self.Q.T[:-1], np.ones(n)])
        b = np.zeros(n)
        b[-1] = 1

        try:
            pi = np.linalg.solve(A.T @ A, A.T @ b)
        except np.linalg.LinAlgError:
            pi = np.linalg.lstsq(A, b, rcond=None)[0]

        pi = np.real(pi)
        pi[pi < 0] = 0
        pi /= pi.sum()

        return {s: pi[i] for i, s in enumerate(self.states)}

    def expected_visits(self, initial_state: str, t: float) -> Dict[str, float]:
        """Calculate expected number of visits to each state by time t"""
        self._validate_initial_state(initial_state)
        idx = self.state_indices[initial_state]
        integral = np.linalg.inv(self.Q) @ (expm(self.Q * t) - np.eye(len(self.states)))
        return {s: integral[idx, i] for i, s in enumerate(self.states)}

    def time_average_distribution(self, t: float, initial_state: str) -> Dict[str, float]:
        """Compute time-averaged distribution up to time t"""
        visits = self.expected_visits(initial_state, t)
        return {s: visits[s]/t for s in self.states}

    def plot_state_transitions(self, trajectory: List[tuple], figsize: tuple = (10, 4),
                               save_path: Optional[str] = None) -> None:
        """
        Visualize state transitions over time using step plot
        :param:trajectory: List of (time, state) tuples from generate_trajectory()
        :param:figsize: Figure dimensions (width, height) in inches
        :param:save_path: Optional path to save the plot image
        :returns:None
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Visualization requires matplotlib. Install via 'pip install matplotlib'")

        times, states = zip(*trajectory)
        state_labels = sorted(set(states), key=self.states.index)  # Preserve order
        numeric_states = [state_labels.index(s) for s in states]

        plt.figure(figsize=figsize)
        plt.step(times, numeric_states, where='post')
        plt.yticks(range(len(state_labels)), state_labels)
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.title("Continuous-Time Markov Process State Transitions")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class InfiniteContinuousTimeMarkovProcess:
    """
    Continuous-time Markov process with infinite state space

    :param transition_rates: Function (i) -> Dict[int, float] returning transition rates from state i
    """

    def __init__(self, transition_rates: Callable[[int], Dict[int, float]]):
        self.transition_rates = transition_rates
        self.current_state = None

    def jump_process(self, initial_state: int, max_jumps: int = 1000) -> List[tuple]:
        """Simulate jump process with (time, state) sequence"""
        sequence = [(0.0, initial_state)]
        current_state = initial_state
        current_time = 0.0

        for _ in range(max_jumps):
            rates = self.transition_rates(current_state)
            total_rate = sum(rates.values())

            if total_rate == 0:
                break  # Absorbing state

            dwell_time = np.random.exponential(1/total_rate)
            current_time += dwell_time

            next_state = np.random.choice(
                list(rates.keys()),
                p=[r/total_rate for r in rates.values()]
            )

            sequence.append((current_time, next_state))
            current_state = next_state

        return sequence

    def generator_matrix(self, max_states: int) -> np.ndarray:
        """Construct truncated generator matrix"""
        Q = np.zeros((max_states, max_states))

        for i in range(max_states):
            rates = self.transition_rates(i)
            total_rate = sum(rates.values())
            Q[i,i] = -total_rate

            for j, r in rates.items():
                if j < max_states:
                    Q[i,j] = r

        return Q

    def approximate_stationary(self, max_states: int = 100) -> Dict[int, float]:
        """Approximate stationary distribution by truncating state space"""
        Q = self.generator_matrix(max_states)
        ctmp = ContinuousTimeMarkovProcess(Q, list(range(max_states)))
        return ctmp.stationary_distribution()
