import numpy as np
from numpy import sqrt

def const(m):
    #  Returns two constellation in symbols and integers (from 0 to m-1)
    #  m - modulation order
    p = int(sqrt(m))  # (p,p) constellation
    d1 = np.arange(-p + 1, p, 2)
    d2 = np.flip(d1) * 1j
    a1, a2 = np.meshgrid(d1, d2)
    const_sym = (a1 + a2) / sqrt(2 / 3 * (m - 1))  # constellation matrix in symbols
    const_ints = np.arange(0, m).reshape((p, p), order='F')  # constellation matrix in bits
    return const_sym, const_ints


class QAMmodulator:
    """
    Class of QAM modulator
    _________
    Attribute
    m is modulation order (QPSK modulation by default)
    ________
    Method
    modulate(bits) - modulate bits to symbols
    """
    def __init__(self, m=4):
        if m not in [4, 16, 64]:
            raise ValueError('Choose m from [4, 16 ,64]')
        self.m = m

    def modulate(self, bits):
        k = int(np.log2(self.m))  # bits per symbol
        if bits.size % k !=0:
            raise ValueError('The length of the bit stream must be a multiple of log2(m)!')
        u = bits.reshape((-1, k))    # k bits in every row
        for i in range(k):
            u[:, i] = u[:, i] * 2 ** (k - i - 1)  # preparation transform bits to integers
        c_s, c_i = const(self.m)
        ints = np.sum(u, axis=1)  # bits to integers
        symb = np.array([c_s[np.where(c_i == i)] for i in ints])  # modulation
        return symb.reshape((1, -1))


class QAMdemodulator:
    """
    Class of QAM demodulator
    _________
    Attribute
    m is modulation order (QPSK modulation by default)
    ________
    Method
    demodulate(sym) - demodulate symbols to bits
    """

    def __init__(self, m=4):
        if m not in [4, 16, 64]:
            raise ValueError('Choose m from [4, 16 ,64]')
        self.m = m

    def demodulate(self, sym):
        k = int(np.log2(self.m))  # bits per symbol
        c_s, c_i = const(self.m)
        z_demod = np.zeros_like(sym)
        for i in range(np.size(sym)):
            dist = np.abs(c_s-sym[0, i])  # find distances to constellation points
            z_demod[0, i] = c_s[np.where(dist == np.min(np.min(dist)))][0]
        demod = np.array([c_i[np.where(c_s == i)] for i in z_demod[0]]).reshape((1, -1))  # demodulate to integer representation
        s = ''.join([(np.binary_repr(i, width=k)) for i in demod[0]])
        bits = np.array([int(symbol) for symbol in s]).reshape((1, -1))  # demodulate to bit representation
        return bits
