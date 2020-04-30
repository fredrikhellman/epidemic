import numpy as np
import matplotlib.pyplot as plt

import epidemic_compute

# Expected duration of disease in days
lam = 8
t_low = 40
t_normal = 300
t_max = 600

m = 100
REs = lam*np.linspace(0.25, 0.1, m)
R0 = 2.4
pop = 1000000

ts, xs, ys, zs = epidemic_compute.simulate(lam, R0, REs, t_low, t_normal, t_max, pop)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

plotm = np.arange(m)[0:-1:8]

colors = [np.array([1, 1, 1])*(i/(m-1)*0.0 + (m-1-i)/(m-1)*0.8) for i in plotm]

[ax1.semilogy(ts, y, ls=':', color=color) for (y, color) in zip(np.array(ys).T[plotm], colors)]
[ax1.semilogy(ts, z, ls='-', color=color, label='RE(låg)={:.2}'.format(r*lam)) for (r, z, color) in zip(REs[plotm], np.array(zs).T[plotm], colors)]

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

ax2.plot(REs, np.array(zs)[-1,:], label=f'Antal immuna efter {t_max} dagar')
ax2.plot(REs, np.max(np.array(ys), axis=0), label=f'Maximalt antal sjuka samtidigt')
ax2.grid(which='major', axis='both', linestyle='--')
the_herd = pop*(1-1/(R0))
ax2.plot(REs, len(REs)*[the_herd], '--', color='gray')
ax2.annotate(f'Gräns för flockimmunitet under R0',
             (REs[-1], the_herd))

ax2.set_xlabel('RE(låg)')
ax2.set_ylabel(f'Antal personer (befolkning = 1 milj.)')
ax2.legend()

plt.savefig('graf_family.png', )


plt.show()
    
