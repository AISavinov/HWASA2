from  sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

X = np.array([3,5,7,9,9,10,11,13,17,19]).reshape(-1, 1)
x = [3,5,7,9,9,10,11,13,17,19]
N = 100
np.random.seed(1)
X_plot = np.linspace(-5, 20, 1000)[:, np.newaxis]

from collections import Counter
fig, ax = plt.subplots()
n, bins, patches = ax.hist(X, density=1)
c = Counter(x)
y = c.values()
#ax.plot(bins, y, '--')

colors = ['navy', 'cornflowerblue', 'darkorange']
kernels = ['gaussian', 'epanechnikov']
lw = 2

for color, kernel in zip(colors, kernels):
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), color=color, lw=lw,
            linestyle='-', label="kernel = '{0}'".format(kernel))

ax.legend(loc='upper left')
ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

ax.set_xlim(2, 20)
ax.set_ylim(-0.02, 1)
plt.show()

from  sklearn.neighbors import KernelDensity
import numpy as np
X = np.array([3,5,7,9,9,10,11,13,17,19])

gaussian = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X.reshape(-1, 1))
print(np.exp(gaussian.score_samples(np.array([5, 10]).reshape(-1, 1))))

epanechnikov = KernelDensity(kernel='epanechnikov', bandwidth=0.5).fit(X.reshape(-1, 1))
print(np.exp(epanechnikov.score_samples(np.array([5, 10]).reshape(-1, 1))))
