import numpy as np
import matplotlib.pyplot as plt
from .MarkovChain import MarkovChain

class BirthDeathChain(MarkovChain):
    """
    Realization of birth death chain
    Inherits from Markovchain class and automatically constructs tridiagonal transition matrix

    Features:
    -The state space is {0, 1, 2,..., n}
    -Can only transfer to adjacent state or keep current state
    -Boundary state processing (state 0 cannot die, state n cannot be born)

    : param birth_rates:  Birth rate list (length n), birth_rates[i] indicates the probability from state I to i+1
    : param death_rates:  Mortality list (length n), death_rates[i] indicates the probability from state i+1 to I
    : param states:  Optional status label (default is digital status)
    """
    def __init__(self, birth_rates, death_rates, states=None):
        self._validate_rates(birth_rates, death_rates)

        N = len(birth_rates) + 1
        self.N = N
        self.birth = birth_rates
        self.death = death_rates

        P = self._build_transition_matrix()

        if states is None:
            states = [str(i) for i in range(N)]

        super().__init__(P, states)

    def _validate_rates(self, birth_rates, death_rates):
        if len(death_rates) != len(birth_rates):
            raise ValueError("Birth_rates and death_rates must be the same length")

        if any(r < 0 for r in birth_rates + death_rates):
            raise ValueError("Birth and death rate cannot be negative")
        if len(birth_rates) < 1:
            raise ValueError("At least 1 birth rate is required")
        if any(b > 1 for b in birth_rates):
            raise ValueError("Birth rate cannot exceed 1")

        for i, (b, d) in enumerate(zip(birth_rates[:-1], death_rates[1:])):
            if b + d > 1:
                raise ValueError(f"The total transition probability of state {i+1} exceeds 1: birth[{i}]={b}, death[{i+1}]={d}")

    def _build_transition_matrix(self):
        N = self.N
        P = np.zeros((N, N))

        for i in range(N):
            if i == 0:
                birth_prob = self.birth[0]
                stay_prob = 1 - birth_prob
                P[i][i+1] = birth_prob
                P[i][i] = stay_prob
            elif i == N-1:
                death_prob = self.death[-1]
                stay_prob = 1 - death_prob
                P[i][i-1] = death_prob
                P[i][i] = stay_prob
            else:
                birth_prob = self.birth[i]
                death_prob = self.death[i-1]
                stay_prob = 1 - (birth_prob + death_prob)

                if stay_prob < 0:
                    raise ValueError(f"The total transition probability of state {i} exceeds 1: birth={birth_prob}, death={death_prob}")

                P[i][i+1] = birth_prob
                P[i][i-1] = death_prob
                P[i][i] = stay_prob

        return P

    def analytic_stationary(self):
        """
        Calculating the analytic steady state distribution of birth and death processes
        Formula: π ˊ u i=π ˊ u 0 * Πˊ u {k=0} ^ {i-1} (birth ˊ U K/death ˊ u {k+1})
        """
        pi0 = 1.0
        product = 1.0

        for i in range(1, self.N):
            product *= self.birth[i - 1] / self.death[i - 1]
            pi0 += product

        pi0 = 1 / pi0

        pi = [pi0]
        current = pi0
        for i in range(1, self.N):
            current *= (self.birth[i-1] + 1e-12) / (self.death[i-1] + 1e-12)
            pi.append(current)

        return {s: pi[i] for i, s in enumerate(self.states)}

    def plot_stationary(self):
        """
        Draw a comparison diagram of steady-state distribution
        """

        numeric = self.stationary_distribution()
        analytic = self.analytic_stationary()

        plt.bar(range(self.N), [numeric[s] for s in self.states], alpha=0.6, label='数值解')
        plt.plot(range(self.N), [analytic[s] for s in self.states], 'ro--', label='解析解')
        plt.xticks(range(self.N), self.states)
        plt.title('Steady state distribution comparison')
        plt.legend()
        plt.show()
