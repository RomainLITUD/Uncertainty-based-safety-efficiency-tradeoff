import numpy as np
from scipy.integrate import simps

def Entropy_simpson(H):
    e = np.zeros(len(H))
    x = np.linspace(-22.5,23,92)
    y = np.linspace(-11.5,75,174)
    
    for i in range(len(H)):
        Hp = H[i]/np.sum(H)
    