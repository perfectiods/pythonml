import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt

# via https://proglib.io/p/neural-nets-guide

# define sigmoid
x = np.arange(-8, 8, 0.1)
f = 1 / (1 + np.exp(-x))
plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

# weighted enter in node:
# x1w1+x2w2+x3w3+b, where b - smesheniye
# by changing b we can regulate nodes activation time
# let b = 1


