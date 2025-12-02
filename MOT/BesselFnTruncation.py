import numpy as np
import scipy
import scipy.special as fns
import matplotlib.pyplot as plt

# Check where the sum over Besselfunctions can be truncated
# Following rule of thumb is given: 
# Bessel function decay exponentially when their index exceeds their argument

def f(arg):
    return lambda index: fns.hankel2(index, arg)

indices = np.arange(24)
arguments = np.linspace(1, 18, 5)

for arg in arguments:
    F = f(arg)(indices)
    plt.plot(indices, F, label=f"arg={arg}")
    # plt.plot(indices, F[int(np.ceil(arg))]*np.exp((np.ceil(arg)-indices)/((arg + 9)/13.5)))
plt.xlabel("n")
plt.ylim(-.5,.5)
plt.legend()
plt.show()

# Rule of thumb is indeed visible.

# Exponential decay: when negligable?
x = np.linspace(0, 3, 32)
plt.plot(x, np.exp(-x))
plt.show()

# at about 2 already only 1 to 2%