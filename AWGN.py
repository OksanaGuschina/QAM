import numpy as np

class Awgn:
    """
    Class of additive white gaussian noise
    _________
    Attribute
    var is variance of noise (var = 1 by default)
    ________
    Method
    generate_noise(N) - create row vector with shape (1,N) of complex random variables with normal distribution

    """
    def __init__(self, var=1):
        self.var = var

    def generate_noise(self, N):
        return np.sqrt(self.var / 2) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))