# Author: Sequoia Ploeg

import numpy as np
import matplotlib.pyplot as plt

def removeGC(amp_raw, wl):
    POLY_ORDER = 4
    p = np.polyfit((wl - np.mean(wl)), amp_raw, POLY_ORDER)
    amplitude_baseline = np.polyval(p, wl - np.mean(wl))
    
    amplitude_corrected = amp_raw - amplitude_baseline
    amplitude_corrected = amplitude_corrected + np.max(amplitude_baseline) - np.max(amp_raw)
    return amplitude_corrected

def linear2db(power):
    return 10 * np.log10(power)

data = np.load('data.npz')
wl = data['wavelength']
power = data['power']

start, end = np.argmax(wl > 1560), np.argmax(wl > 1620)
wl = wl[start:end]
power = power[start:end]

power[np.where(power <= 0)] = 1e-3
power = linear2db(power)
power = removeGC(power, wl)
power = 10 ** (power / 10)

plt.plot(wl, power)
plt.show()
