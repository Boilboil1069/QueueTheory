import numpy as np


class MarkovChain:
    def __init__(self, transition_matrix: np.ndarray, states: list,stationary_method='power'):
        """
        :param transition_matrix: 状态转移概率矩阵（n x n）
        :param states: 状态标签列表（长度n）
        """
        # 参数校验
        if not np.allclose(transition_matrix.sum(axis=1), 1):
            raise ValueError("转移矩阵每行概率和必须为1")
        if transition_matrix.min() < 0:
            raise ValueError("转移概率不能为负数")
        if len(states) != transition_matrix.shape[0]:
            raise ValueError("状态标签数量与转移矩阵维度不匹配")

        self.P = transition_matrix.astype(np.float64)  # 转移矩阵
        self.states = states  # 状态标签
        self.state_indices = {s: i for i, s in enumerate(states)}  # 状态索引映射
        self.current_state = None
        self.stationary_method = stationary_method

    def set_initial(self, state):
        """设置初始状态"""
        if state not in self.state_indices:
            raise ValueError("初始状态不存在")
        self.current_state = self.state_indices[state]

    def next_state(self):
        """转移到下一状态"""
        if self.current_state is None:
            raise RuntimeError("未设置初始状态")
        next_idx = np.random.choice(
                len(self.states),
                p=self.P[self.current_state]
        )
        self.current_state = next_idx
        return self.states[next_idx]

    def generate_sequence(self, steps: int, initial_state=None):
        """
        生成状态序列
        :param steps: 转移步数
        :param initial_state: 初始状态（可选）
        :return: 状态序列列表
        """
        if initial_state is not None:
            self.set_initial(initial_state)
        elif self.current_state is None:
            raise RuntimeError("需要指定初始状态")

        sequence = [self.states[self.current_state]]
        for _ in range(steps):
            sequence.append(self.next_state())
        return sequence

    def _stationary_distribution_power(self, max_iter=1000, tol=1e-6):
        """计算稳态分布（幂迭代法）"""
        n = len(self.states)
        pi = np.ones(n) / n

        for _ in range(max_iter):
            new_pi = pi @ self.P
            if np.linalg.norm(new_pi - pi) < tol:
                break
            pi = new_pi
        return {s: pi[i] for i, s in enumerate(self.states)}

    def _stationary_distribution_eigen(self, tol=1e-9):
        """使用特征向量法计算稳态分布（适用于可逆链）"""

        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)

        close_to_one = np.isclose(eigenvalues, 1, atol=tol)
        if not np.any(close_to_one):
            raise ValueError("未找到特征值为1的特征向量，可能不存在唯一稳态分布")

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
            raise ValueError("存在负概率值，可能不是有效稳态分布")
        pi[pi < 0] = 0

        pi /= pi.sum()
        return {s: pi[i] for i, s in enumerate(self.states)}

    def _stationary_distribution_linear(self, tol=1e-10):
        """通过求解线性方程组 (P^T - I)π = 0 且 Σπ=1 计算稳态分布"""
        n = len(self.states)

        A = np.zeros((n, n))
        A[:-1, :] = (self.P.T - np.eye(n))[:-1]
        A[-1, :] = 1

        # 构造右侧向量
        b = np.zeros(n)
        b[-1] = 1

        # 求解线性方程组
        try:
            pi = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 使用最小二乘法处理秩不足的情况
            pi, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
            if rank < n:
                raise ValueError("系数矩阵秩不足，无法获得唯一解")

        # 数值稳定性处理
        pi = np.real(pi)  # 确保实数解
        pi[pi < 0] = 0  # 清除微小负值
        pi /= pi.sum()  # 重新归一化

        if not self.verify_stationary(dict(zip(self.states, pi)), tol):
            raise RuntimeError("线性方程组法求解结果不满足稳态条件")

        A_reg = A.T @ A + tol * np.eye(n)
        b_reg = A.T @ b
        pi = np.linalg.solve(A_reg, b_reg)

        # 添加残差检查
        residual = np.linalg.norm(A @ pi - b)
        if residual > 1e-6:
            raise RuntimeError(f"线性方程组求解残差过大：{residual:.2e}")

        return {s: pi[i] for i, s in enumerate(self.states)}


    def stationary_distribution(self, method='power', **kwargs):
        if method == 'power':
            return self._stationary_distribution_power(**kwargs)
        elif method == 'eigen':
            return self._stationary_distribution_eigen(**kwargs)
        elif method == 'linear':
            return self._stationary_distribution_linear(**kwargs)
        else:
            raise ValueError("不支持的method参数，可选：'power', 'eigen'")


    def transition_count(self, sequence):
        """
        统计状态转移频次
        """
        count = np.zeros_like(self.P)
        for i in range(len(sequence) - 1):
            from_idx = self.state_indices[sequence[i]]
            to_idx = self.state_indices[sequence[i + 1]]
            count[from_idx, to_idx] += 1
        return count

    def verify_stationary(self, pi, tol=1e-6):
        """
        验证分布是否满足πP=π
        """
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


