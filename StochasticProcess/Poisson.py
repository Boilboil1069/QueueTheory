import numpy as np

class PoissonProcess:
    def __init__(self, rate):
        """
        :param rate: 事件发生率（λ>0）
        """
        if rate <= 0:
            raise ValueError("Rate must be positive")
        self.rate = float(rate)

    def generate_events(self, T):
        """
        生成时间区间[0,T]内的事件发生时间序列
        :param T: 时间区间长度
        :return: 事件发生时间数组（升序排列）
        """
        intervals = []
        total_time = 0.0

        # 生成事件间隔时间（指数分布）
        while total_time < T:
            dt = np.random.exponential(1 / self.rate)
            total_time += dt
            if total_time < T:
                intervals.append(total_time)

        return np.array(intervals)

    def event_count(self, t):
        """
        计算时间t内的事件数量（泊松分布）
        :param t: 时间长度
        :return: 事件数（随机变量）
        """
        return np.random.poisson(self.rate * t)

    def expectation(self, t):
        """
        时间t内事件数的期望值
        :param t: 时间长度
        :return: E[N(t)] = λt
        """
        return self.rate * t

    def variance(self, t):
        """
        时间t内事件数的方差
        :param t: 时间长度
        :return: Var[N(t)] = λt
        """
        return self.rate * t

    def time_between_events(self, size=1):
        """
        生成事件间隔时间（指数分布）
        :param size: 生成的数量
        :return: 间隔时间数组
        """
        return np.random.exponential(1 / self.rate, size)
