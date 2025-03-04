import numpy as np
import matplotlib.pyplot as plt
from math import exp, isclose
from MarkovChain import MarkovChain

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


class InfiniteBirthDeathChain(MarkovChain):
    """
    Infinite-state birth-death process chain implementation

    Features:
    - Dynamic state space truncation for numerical solutions
    - Two solving methods: analytic and numerical approximation
    - Transient analysis and visualization
    - Auto-optimized state space truncation

    :param birth_func: Function returning birth probability for state i (i -> i+1)
    :param death_func: Function returning death probability for state i (i -> i-1)
    :param max_truncate: Maximum allowed truncated states (default 500)
    """

    def __init__(self, birth_func, death_func, max_truncate=500):
        self.birth = birth_func
        self.death = death_func
        self.max_truncate = max_truncate
        self._verify_stability()

        P, states = self._build_truncated_matrix()
        super().__init__(P, states)

    def _verify_stability(self, eps=1e-6):
        """Verify process stability using convergence criteria"""
        product = 1.0
        sum_pi = 1.0

        for i in range(1, 1000):  # Check first 1000 terms
            try:
                rho = self.birth(i - 1) / self.death(i)
            except ZeroDivisionError:
                raise ValueError("Zero death rate detected")

            product *= rho
            sum_pi += product

            if rho >= 1.0:
                raise RuntimeError("Process does not satisfy stability condition ρ < 1")

            if product < eps:
                break

        if self.birth(0) / self.death(1) >= 1.0:
            raise RuntimeError("Stability condition violated: ρ >= 1")

    def _build_truncated_matrix(self, tolerance=1e-6):
        """Dynamically build transition matrix with auto-truncation"""
        states = ['0']
        P = np.zeros((1, 1))
        P[0, 0] = 1 - self.birth(0)

        for i in range(1, self.max_truncate):
            # Expand matrix to include new state
            new_size = i + 1
            new_P = np.zeros((new_size, new_size))
            new_P[:i, :i] = P  # Copy existing matrix

            # Update transitions for new state
            if i > 0:
                new_P[i, i - 1] = self.death(i)  # Death from i to i-1
            new_P[i, i] = 1 - self.birth(i) - self.death(i)  # Staying probability

            # Add birth transition if not at max size
            if i < new_size - 1:
                new_P[i, i + 1] = self.birth(i)  # Birth to i+1

            # Update previous state's birth transition
            if i > 0:
                new_P[i - 1, i] = self.birth(i - 1)

            # Check if expansion needed
            if self.birth(i) < tolerance:
                break

            P = new_P
            states.append(str(i))

        return P, states

    def analytic_solution(self, tolerance=1e-6):
        """Calculate analytic solution with auto-truncation"""
        pi = [1.0]  # π0
        product = 1.0

        for i in range(1, self.max_truncate):
            try:
                rho = self.birth(i - 1) / self.death(i)
            except ZeroDivisionError:
                break

            product *= rho
            current_pi = pi[0] * product

            if current_pi < tolerance:
                break

            pi.append(current_pi)

        total = sum(pi)
        return {i: p / total for i, p in enumerate(pi)}

    def numerical_solution(self, method='power', **kwargs):
        """Numerical solution using dynamic truncation"""
        return super().stationary_distribution(method=method, **kwargs)

    def transient_analysis(self, initial_state, steps):
        """Calculate transient probabilities after N steps"""
        if isinstance(initial_state, int):
            initial_state = str(initial_state)

        needed_states = min(steps * 2 + int(initial_state), self.max_truncate)
        self._build_truncated_matrix(needed_states)

        return self._transient_matrix_power(initial_state, steps)

    def _transient_matrix_power(self, initial_state, steps):
        """Matrix exponentiation method for transient analysis"""
        state_idx = self.states.index(initial_state)
        initial_vec = np.zeros(len(self.states))
        initial_vec[state_idx] = 1.0

        P_power = np.linalg.matrix_power(self.P, steps)
        result = initial_vec @ P_power
        return {s: result[i] for i, s in enumerate(self.states)}

    def plot_distribution(self, num_states=20, analytic=True, numeric=True):
        """Plot probability distributions"""
        plt.figure(figsize=(12, 6))

        if analytic:
            analytic_pi = self.analytic_solution()
            states = list(analytic_pi.keys())[:num_states]
            probs = [analytic_pi[i] for i in states]
            plt.plot(states, probs, 'ro--', label='Analytic Solution')

        if numeric:
            numeric_pi = self.numerical_solution()
            states = list(map(int, self.states))[:num_states]
            probs = [numeric_pi[s] for s in self.states[:num_states]]
            plt.bar(states, probs, alpha=0.5, label='Numerical Solution')

        plt.xlabel('State')
        plt.ylabel('Probability')
        plt.title('Stationary Distribution Comparison')
        plt.legend()
        plt.show()

    def plot_transient(self, initial_state, max_steps=50):
        """Plot transient probability evolution"""
        time_steps = range(0, max_steps + 1)
        prob_0 = []

        for t in time_steps:
            dist = self.transient_analysis(initial_state, t)
            prob_0.append(dist['0'])

        plt.plot(time_steps, prob_0)
        plt.xlabel('Time Steps')
        plt.ylabel('Probability in State 0')
        plt.title(f'Transient Probability Evolution (Initial State {initial_state})')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    birth = [0.5, 0.4, 0.3]
    death = [0.2, 0.3, 0.4]

    try:
        bdc = BirthDeathChain(birth, death)

        print("转移矩阵：")
        print(bdc.P)

        numeric_pi = bdc.stationary_distribution(method='power')

        analytic_pi = bdc.analytic_stationary()

        print("\n稳态分布对比：")
        print(f"{'状态':<5} {'数值解':<10} {'解析解':<10}")
        for s in bdc.states:
            print(f"{s:<5} {numeric_pi[s]:<10.6f} {analytic_pi[s]:<10.6f}")

        seq = bdc.generate_sequence(10, initial_state='1')
        print("\n生成序列：", seq)

    except ValueError as e:
        print("参数错误：", e)

    # M/M/1 queue parameters
    lam = 0.6  # Arrival rate
    mu = 0.8  # Service rate


    # Define birth-death functions
    def birth_func(i):
        return lam  # Constant arrival rate


    def death_func(i):
        return mu if i > 0 else 0  # Constant service rate


    # Create chain instance
    bd_chain = InfiniteBirthDeathChain(birth_func, death_func)

    # Calculate solutions
    analytic = bd_chain.analytic_solution()
    numeric = bd_chain.numerical_solution(method='linear')

    # Print comparison
    print("State\tAnalytic\tNumeric")
    for i in range(5):
        print(f"{i}\t{analytic.get(i, 0):.4f}\t\t{numeric.get(str(i), 0):.4f}")

    # Visualize distributions
    bd_chain.plot_distribution(num_states=10)

    # Transient analysis example
    bd_chain.plot_transient(initial_state=5, max_steps=20)