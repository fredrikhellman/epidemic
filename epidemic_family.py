import csv
import numpy as np
import re
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.sparse import lil_matrix, csr_matrix

# Expected duration of disease in days
lam = 8

T = 600
m = 100
# Infection rates at time t (days)
r_low = np.linspace(0.25, 0.1, m)
r_normal = m*[0.3]
t_low = 40
t_normal = 300

rlist_of_t = lambda t: (r_normal if t < t_low else
                        r_low if t < t_normal else
                        r_normal)

pop = 1000000
# Population sizes
N = np.array(m*[pop])

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

for n in range(int(np.ceil(T/h))):
    print(n)

    t = h*n

    X = np.column_stack([xs[-1], ys[-1], zs[-1]]).astype(int)
    #print(X)

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


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

plotm = np.arange(m)[0:-1:8]

colors = [np.array([1, 1, 1])*(i/(m-1)*0.0 + (m-1-i)/(m-1)*0.8) for i in plotm]

[ax1.semilogy(ts, y, ls=':', color=color) for (y, color) in zip(np.array(ys).T[plotm], colors)]
[ax1.semilogy(ts, z, ls='-', color=color, label='RE(låg)={:.2}'.format(r*lam)) for (r, z, color) in zip(r_low[plotm], np.array(zs).T[plotm], colors)]

ax1.grid(which='major', axis='both', linestyle='--')
ax1.set_xticks([0, t_low, t_normal, ts[-1]])

ax1.text(t_low/2, 1.5e6, f'normal (R0)', horizontalalignment='center')
ax1.text((t_low+t_normal)/2, 1.5e6, f'låg (RE)')
ax1.text((t_normal+ts[-1])/2, 1.5e6, f'normal (R0)')

ax1.annotate(f'{int(zs[-1][0])}', (ts[-1], zs[-1][0]),
             xytext=(1.05*ts[-1], 3*zs[-1][0]),
             arrowprops={'arrowstyle': '->'})
ax1.annotate(f'{int(zs[-1][-1])}', (ts[-1], zs[-1][-1]),
             xytext=(0.92*ts[-1], 3*zs[-1][-1]),
             arrowprops={'arrowstyle': '->'})
ax1.set_ylim([1, 1e6])
ax1.set_xlabel('Dagar från första fallet')
ax1.set_ylabel('Antal personer (befolkning = 1 milj.)')
ax1.legend()

ax2.plot(r_low*lam, np.array(zs)[-1,:], label=f'Antal immuna efter {T} dagar')
ax2.plot(r_low*lam, np.max(np.array(ys), axis=0), label=f'Maximalt antal sjuka samtidigt')
ax2.grid(which='major', axis='both', linestyle='--')
the_herd = pop*(1-1/(lam*r_normal[0]))
ax2.plot(r_low*lam, len(r_low)*[the_herd], '--', color='gray')
ax2.annotate(f'Gräns för flockimmunitet under R0',
             (r_low[-1]*lam, the_herd))

ax2.set_xlabel('RE(låg)')
ax2.set_ylabel(f'Antal personer som varit sjuka (immuna) efter {T} dagar')
ax2.legend()

plt.savefig('graf_family.png', )


plt.show()
    
