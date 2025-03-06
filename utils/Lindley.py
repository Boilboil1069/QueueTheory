class LindleyEquation:
    """
    A class to compute waiting times using Lindley's equation for G/G/1 queue systems.

    Lindley's equation formula:
    W_{n+1} = max(0, W_n + S_n - A_n)

    where:
    - W_n: waiting time of nth customer
    - S_n: service time of nth customer
    - A_n: inter-arrival time between nth and (n+1)th customer
    """

    def __init__(self, service_times, inter_arrival_times):
        """
        Initialize the Lindley equation solver.

        :param service_times (list): List of service times for each customer [S_0, S_1, ..., S_{n-1}]
        :param inter_arrival_times (list): List of inter-arrival times [A_0, A_1, ..., A_{n-1}]
        """
        if len(service_times) != len(inter_arrival_times):
            raise ValueError("Service times and inter-arrival times must have the same length")

        self.service_times = service_times
        self.inter_arrival_times = inter_arrival_times
        self.waiting_times = [0]  # Initialize with W_0 = 0

    def solve(self):
        """
        Compute waiting times for all customers using Lindley's recursive equation.

        :returns list: List of waiting times [W_0, W_1, ..., W_n]
        """
        for n in range(len(self.service_times)):
            current_wait = self.waiting_times[n]
            next_wait = max(0, current_wait + self.service_times[n] - self.inter_arrival_times[n])
            self.waiting_times.append(next_wait)

        return self.waiting_times

    def get_final_waiting_time(self):
        """
        Get the final waiting time in the system.

        :return float: Last element from the waiting times list
        """
        return self.waiting_times[-1]
