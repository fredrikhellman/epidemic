import csv
import numpy as np
import re
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.sparse import lil_matrix, csr_matrix

# Expected duration of disease in days
lam = 8

m = 2
# Infection rates at time t (days)
r_normal = m*[0.3]
r_low = [0.3, 0.1]
t_low = 40
t_normal = 200

rlist_of_t = lambda t: (r_normal if t < t_low else
                        r0_low if t < t_normal else
                        r_normal)

# Population sizes
N = np.array(m*[1000000])

r_of_t = lambda t: np.array(rlist_of_t(t))

# Time step
h = 0.1

# x - # not yet infected people
# y - # currently infected
# z - # immune

x = N
y = np.ones_like(x)
z = np.zeros_like(x)

xs = [x]
ys = [y]
zs = [z]
ts = [0]

for n in range(int(np.ceil(400/h))):
    print(n)

    t = h*n

    X = np.column_stack([xs[-1], ys[-1], zs[-1]]).astype(int)
    print(X)

    # Compute reaction matrix
    R = np.diag(r_of_t(t))

    # Compute transitions
    hRy = h*np.dot(R, ys[-1]/N)
    x_to_y = xs[-1] - xs[-1]/(1+hRy)
    y_to_z = h/lam*ys[-1]

    # Apply transitions
    xs.append(xs[-1] - x_to_y)
    ys.append(ys[-1] + x_to_y - y_to_z)
    zs.append(zs[-1] + y_to_z)
    ts.append(t)

plt.semilogy(ts, np.array(ys)[:,0], ls=':', color='r')
plt.semilogy(ts, np.array(ys)[:,1], ls=':', color='b')

plt.semilogy(ts, np.array(zs)[:,0], ls='-', color='r')
plt.semilogy(ts, np.array(zs)[:,1], ls='-', color='b')

plt.legend(['Sjuka i region röd',
            'Sjuka i region blå',
            'Immuna eller döda i region röd',
            'Immuna eller döda i region blå'])

ax = plt.gca()
ax.grid(which='major', axis='both', linestyle='--')
ax.set_xticks([0, t_low, t_normal, ts[-1]])

ax.text(t_low/2, 1.5e6, f'r={r_normal}\nlam=8', horizontalalignment='center')
ax.text((t_low+t_normal)/2, 1.5e6, f'r={r0_low} (röd)\nr={r1_low} (blå)\nlam=8')
ax.text((t_normal+ts[-1])/2, 1.5e6, f'r={r_normal}\nlam=8')

plt.annotate(f'{int(zs[-1][0])}', (ts[-1], zs[-1][0]),
             xytext=(1.05*ts[-1], 3*zs[-1][0]),
             arrowprops={'arrowstyle': '->'})
plt.annotate(f'{int(zs[-1][1])}', (ts[-1], zs[-1][1]),
             xytext=(0.92*ts[-1], 3*zs[-1][1]),
             arrowprops={'arrowstyle': '->'})
plt.ylim([1, 1e6])
plt.xlabel('Dagar från första fallet')
plt.ylabel('Antal personer (befolkning = 1 milj.)')

plt.savefig('graf.png')
plt.show()
    
