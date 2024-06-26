import numpy as np
import matplotlib.pyplot as plt
from QAM_ModulatorDemodulator import QAMmodulator, QAMdemodulator
from AWGN import Awgn

#####################################
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] ='stix'
######################################

maxNumErrs = 100  # first stop criterion for BER measurement
maxNumBits = 1e5  # second stop criterion for BER measurement

varV = np.arange(0, 2, 0.1)  # vector of variances
modM = [4, 16, 64]  # list of modulation orders (4 is for QPSK, 16 is for QAM16, 64 is for QAM64)
ber = np.zeros((len(modM), np.size(varV)))  # BER matrix initialization
N = 2400  # number of bits in 1 frame
for i in range(len(modM)):
    mod = QAMmodulator(m=modM[i])
    demod = QAMdemodulator(m=modM[i])
    for j in range(varV.size):
        awgn = Awgn(var=varV[j])
        numErrs = 0
        numBits = 0
        while (numErrs < maxNumErrs) & (numBits < maxNumBits):
            bits = np.random.choice([0, 1], (1, N))  # bits
            sym = mod.modulate(bits)   # modulated symbols
            n = awgn.generate_noise(np.size(sym))  # noise
            symNoisy = sym + n # noisy symbols
            bits_d = demod.demodulate(symNoisy)  # demodulated bits
            numErrs += np.sum(np.logical_xor(bits, bits_d))
            numBits += N
        ber[i, j] = numErrs/numBits

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
for i in range(ber.shape[0]):
    ax.plot(varV, ber[i, :])
    ax.grid(linewidth=0.08)
ax.set(xlabel=r'$\sigma ^2, В^2$', ylabel=r'BER')
ax.legend(['QPSK', 'QAM16', 'QAM64'], loc='upper left')
ax.set_xlim([min(varV), max(varV)])
ax.set_ylim(bottom=np.min(np.min(ber)))
plt.tight_layout()
plt.show()
