import numpy as np
from typing import Dict, List, Callable, Optional
from MarkovProcess import (ContinuousTimeMarkovProcess,
                            InfiniteContinuousTimeMarkovProcess)


class FiniteBirthDeathProcess(ContinuousTimeMarkovProcess):
    """
    Finite state continuous-time birth-death process

    Features:
    - State space {0, 1, 2,..., N}
    - Birth rates λ_i (state i -> i+1)
    - Death rates μ_i (state i -> i-1)
    - Diagonal elements automatically calculated

    :param birth_rates: List of birth rates (length N)
    :param death_rates: List of death rates (length N)
    :param states: Optional state labels
    """

    def __init__(self, birth_rates: List[float], death_rates: List[float], states: List[str] = None):
        self._validate_rates(birth_rates, death_rates)
        N = len(birth_rates) + 1  # Number of states

        self.birth = birth_rates
        self.death = death_rates
        Q = self._build_generator_matrix(N)

        states = states or [str(i) for i in range(N)]
        super().__init__(Q, states)

    def _validate_rates(self, birth_rates, death_rates):
        if len(birth_rates) != len(death_rates):
            raise ValueError("birth_rates and death_rates must have same length")

        if any(r < 0 for r in birth_rates + death_rates):
            raise ValueError("Rates must be non-negative")

    def _build_generator_matrix(self, N):
        Q = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            # Birth transitions (except last state)
            if i < N-1:
                Q[i, i+1] = self.birth[i]

            # Death transitions (except first state)
            if i > 0:
                Q[i, i-1] = self.death[i-1]

            # Diagonal element
            Q[i,i] = -np.sum(Q[i])

        return Q

    def analytic_stationary(self) -> Dict[str, float]:
        """Compute stationary distribution using birth-death formula"""
        pi = [1.0]
        for i in range(1, len(self.states)):
            product = np.prod([self.birth[k]/self.death[k] for k in range(i)])
            pi.append(pi[0] * product)

        # Normalize
        total = sum(pi)
        return {s: pi[i]/total for i, s in enumerate(self.states)}


class InfiniteBirthDeathProcess(InfiniteContinuousTimeMarkovProcess):
    """
    Infinite state continuous-time birth-death process

    :param birth_fn: Function(i) returning birth rate λ_i
    :param death_fn: Function(i) returning death rate μ_i
    """

    def __init__(self, birth_fn: Callable[[int], float],
                 death_fn: Callable[[int], float]):
        self.birth_fn = birth_fn
        self.death_fn = death_fn

        if death_fn(0) != 0:
            raise ValueError("μ_0 must be zero (no death from state 0)")

        super().__init__(self._transition_rates)

    def _transition_rates(self, i: int) -> Dict[int, float]:
        rates = {}
        if i >= 0:
            rates[i+1] = self.birth_fn(i)  # Birth
        if i > 0:
            rates[i-1] = self.death_fn(i)  # Death
        return rates

    def analytic_stationary(self, max_states=1000) -> Dict[int, float]:
        """Compute truncated stationary distribution"""
        pi = [1.0]
        product = 1.0

        for i in range(1, max_states):
            ratio = self.birth_fn(i-1) / self.death_fn(i)
            product *= ratio
            pi.append(pi[0] * product)

            if pi[-1] < 1e-12:  # Early termination
                break

        total = sum(pi)
        return {i: p/total for i, p in enumerate(pi)}

if __name__ == "__main__":
    # 有限状态示例
    birth = [2.0, 1.5, 0.5]
    death = [1.0, 2.0, 1.0]
    fbdp = FiniteBirthDeathProcess(birth, death)
    print(fbdp.stationary_distribution())

    # 无限状态示例
    def λ(i): return 2.0
    def μ(i): return 1.0 if i > 0 else 0
    ibdp = InfiniteBirthDeathProcess(λ, μ)
    print(ibdp.generator_matrix(10))
    print(ibdp.approximate_stationary())