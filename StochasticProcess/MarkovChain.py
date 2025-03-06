import numpy as np
from typing import Callable, Optional
from scipy.integrate import quad

class MarkovChain:
    def __init__(self, transition_matrix: np.ndarray, states: list, stationary_method='power'):
        """
        :param transition_matrix: state transition probability matrix（n x n）
        :param states: Status label list（Length n）
        """
        # 参数校验
        if not np.allclose(transition_matrix.sum(axis=1), 1):
            raise ValueError("The sum of probabilities per row of the transition matrix must be 1")
        if transition_matrix.min() < 0:
            raise ValueError("Transition probability cannot be negative.")
        if len(states) != transition_matrix.shape[0]:
            raise ValueError("The number of status tags does not match the dimension of the transition matrix.")

        self.P = transition_matrix.astype(np.float64)  # 转移矩阵
        self.states = states  # 状态标签
        self.state_indices = {s: i for i, s in enumerate(states)}  # 状态索引映射
        self.current_state = None
        self.stationary_method = stationary_method

    def _set_initial(self, state):
        if state not in self.state_indices:
            raise ValueError("Initial state does not exist")
        self.current_state = self.state_indices[state]

    def _next_state(self):
        if self.current_state is None:
            raise RuntimeError("Initial state not set")
        next_idx = np.random.choice(
                len(self.states),
                p=self.P[self.current_state]
        )
        self.current_state = next_idx
        return self.states[next_idx]

    def generate_sequence(self, steps: int, initial_state=None):
        """
        Generate status sequence.
        :param steps: Transfer steps
        :param initial_state: Initial state (optional)
        :return: Status sequence list
        """
        if initial_state is not None:
            self._set_initial(initial_state)
        elif self.current_state is None:
            raise RuntimeError("Initial state needs to be specified")

        sequence = [self.states[self.current_state]]
        for _ in range(steps):
            sequence.append(self._next_state())
        return sequence

    def _stationary_distribution_power(self, max_iter=1000, tol=1e-6):
        """Calculation of steady state distribution (power iteration method)."""
        n = len(self.states)
        pi = np.ones(n) / n

        for _ in range(max_iter):
            new_pi = pi @ self.P
            if np.linalg.norm(new_pi - pi) < tol:
                break
            pi = new_pi
        return {s: pi[i] for i, s in enumerate(self.states)}

    def _stationary_distribution_eigen(self, tol=1e-9):
        """Calculation of steady-state distribution using eigenvector method (for reversible chains."""

        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)

        close_to_one = np.isclose(eigenvalues, 1, atol=tol)
        if not np.any(close_to_one):
            raise ValueError("The eigenvector with eigenvalue 1 is not found, and there may not be a unique steady-state distribution.")

        idx = np.argmax(close_to_one)
        pi = eigenvectors[:, idx].real

        pi = np.abs(pi)
        pi /= pi.sum()

        real_eigenvectors = eigenvectors[:, np.isreal(eigenvalues)].real
        real_eigenvalues = eigenvalues[np.isreal(eigenvalues)].real
        close_idx = np.argmin(np.abs(real_eigenvalues - 1))
        pi = real_eigenvectors[:, close_idx]

        pi = np.abs(pi)
        if np.any(pi < -tol):
            raise ValueError("There is a negative probability value, which may not be an effective steady-state distribution.")
        pi[pi < 0] = 0

        pi /= pi.sum()
        return {s: pi[i] for i, s in enumerate(self.states)}

    def _stationary_distribution_linear(self, tol=1e-10):
        """
        The steady-state distribution is calculated by solving the system of linear equations (P ^ t - I) π=0 and Σπ=1.
        """
        n = len(self.states)

        A = np.zeros((n, n))
        A[:-1, :] = (self.P.T - np.eye(n))[:-1]
        A[-1, :] = 1

        b = np.zeros(n)
        b[-1] = 1

        try:
            pi = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            pi, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
            if rank < n:
                raise ValueError("The rank of the coefficient matrix is insufficient to obtain a unique solution.")

        pi = np.real(pi)
        pi[pi < 0] = 0
        pi /= pi.sum()

        if not self.verify_stationary(dict(zip(self.states, pi)), tol):
            raise RuntimeError("The solution result of linear equations method does not meet the steady-state condition")

        A_reg = A.T @ A + tol * np.eye(n)
        b_reg = A.T @ b
        pi = np.linalg.solve(A_reg, b_reg)

        residual = np.linalg.norm(A @ pi - b)
        if residual > 1e-6:
            raise RuntimeError(f"The residual error of solving linear equations is too large：{residual:.2e}")

        return {s: pi[i] for i, s in enumerate(self.states)}

    def stationary_distribution(self, method='power', **kwargs):
        if method == 'power':
            return self._stationary_distribution_power(**kwargs)
        elif method == 'eigen':
            return self._stationary_distribution_eigen(**kwargs)
        elif method == 'linear':
            return self._stationary_distribution_linear(**kwargs)
        else:
            raise ValueError("Unsupported method parameter, optional: ' power', 'eigen', 'linear'")

    def transition_count(self, sequence):
        count = np.zeros_like(self.P)
        for i in range(len(sequence) - 1):
            from_idx = self.state_indices[sequence[i]]
            to_idx = self.state_indices[sequence[i + 1]]
            count[from_idx, to_idx] += 1
        return count

    def verify_stationary(self, pi, tol=1e-6):
        pi_array = np.array([pi[s] for s in self.states])
        diff = np.linalg.norm(pi_array @ self.P - pi_array)
        return diff < tol

    def compare_methods(self):

        pi_power = self._stationary_distribution_power()
        pi_eigen = self._stationary_distribution_eigen()
        pi_linear = self._stationary_distribution_linear()

        print("Compare：")
        print(f"{'Status':<10} {'Power':<10} {'Eigen':<10} {'Linear':<10}")
        for s in self.states:
            print(f"{s:<10} {pi_power[s]:<10.6f} {pi_eigen[s]:<10.6f} {pi_linear[s]:<10.6f}")

        max_diff = max(
                max(abs(pi_power[s] - pi_eigen[s]) for s in self.states),
                max(abs(pi_eigen[s] - pi_linear[s]) for s in self.states)
        )
        print(f"\nMax difference：{max_diff:.2e}")

class InfiniteMarkovChain:
    def __init__(self, transition_kernel: Callable[[float], Callable[[float], float]], initial_state: Optional[float] = None):
        self.transition_kernel = transition_kernel
        self.current_state = initial_state
        self.state_history = []
        if initial_state is not None:
            self.state_history.append(initial_state)

    def _set_initial(self, state: float):
        """Set initial state for the chain"""
        self.current_state = state
        self.state_history = [state]

    def _next_state(self):
        """Generate next state using transition kernel"""
        if self.current_state is None:
            raise RuntimeError("Initial state not configured")
        kernel = self.transition_kernel(self.current_state)
        return kernel(np.random.random())

    def generate_sequence(self, steps: int, initial_state=None):
        """Generate state sequence with given steps"""
        if initial_state is not None:
            self._set_initial(initial_state)
        elif self.current_state is None:
            raise RuntimeError("Initial state required")

        sequence = [self.current_state]
        for _ in range(steps):
            next_state = self._next_state()
            sequence.append(next_state)
            self.current_state = next_state
        return sequence

    def transition_density(self, x: float, y: float):
        """Get transition density from state x to y"""
        return self.transition_kernel(x)(y)

    def empirical_distribution(self, bins: int = 100, samples: int = None):
        """Estimate empirical distribution from history"""
        if not self.state_history:
            raise ValueError("No historical states available")

        data = self.state_history[-samples:] if samples else self.state_history
        hist, edges = np.histogram(data, bins=bins, density=True)
        return hist, edges

    def kernel_check(self, x: float, tol: float = 1e-5):
        """Verify kernel integrates to 1 for given state"""
        integral = np.trapezoid([self.transition_density(x, y)
                           for y in np.linspace(0,1,1000)])
        if not np.isclose(integral, 1, atol=tol):
            raise ValueError(f"Kernel violation: Integrates to {integral:.4f}")

    def has_stationary_distribution(self, tol: float = 1e-4, test_samples: int = 10000) -> bool:
        hist1, _ = self.empirical_distribution(bins=100, samples=test_samples // 2)
        hist2, _ = self.empirical_distribution(bins=100, samples=test_samples)
        return np.linalg.norm(hist1 - hist2) < tol

    def stationary_distribution(self, method: str = 'empirical', max_iter: int = 1000, tol: float = 1e-6) -> Callable[[float], float]:
        if method == 'empirical':
            return self._empirical_stationary()
        elif method == 'power':
            return self._power_iteration_stationary(max_iter, tol)
        raise ValueError("Supported methods: 'empirical', 'power'")

    def _empirical_stationary(self) -> Callable[[float], float]:
        hist, edges = self.empirical_distribution()
        return lambda x: hist[np.clip(np.digitize(x, edges) - 1, 0, len(hist) - 1)]

    def _power_iteration_stationary(self, max_iter: int, tol: float) -> Callable[[float], float]:
        pi_prev = lambda x: 1.0
        for _ in range(max_iter):
            pi_next = lambda y: quad(lambda x: pi_prev(x) * self.transition_density(x, y), 0, 1)[0]
            diff = quad(lambda x: abs(pi_next(x) - pi_prev(x)), 0, 1)[0]
            if diff < tol:
                return pi_next
            pi_prev = pi_next
        raise RuntimeError(f"Failed to converge after {max_iter} iterations")

    def verify_stationary(self, pi: Callable[[float], float], tol: float = 1e-4) -> bool:
        _, edges = self.empirical_distribution()

        y_samples = np.clip(np.random.random(100), edges[0], edges[-1] - 1e-6)

        errors = []
        for y in y_samples:
            integral = quad(lambda x: pi(x) * self.transition_density(x, y), edges[0], edges[-1])[0]
            errors.append(abs(pi(y) - integral))

        return np.max(errors) < tol