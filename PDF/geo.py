from utils import numeric


class geo(numeric):
    def __init__(self, p: float,  var: str, interval: tuple):
        """
            :param p: The probability of success
            :param var: The variable of integration
            :param interval: The upper and lower bounds of integration
            """
        def geometric_pdf(i):
            return (1 - p) ** (i - 1) * p

        super().__init__(geometric_pdf, var, interval)
