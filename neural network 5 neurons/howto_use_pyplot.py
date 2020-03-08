import matplotlib.pyplot as plt

#define sigmoid
x = np.arange(-8, 8, 0.1)
f = 1 / (1 + np.exp(-x))
plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)')
#draw sigmoid
plt.show()
