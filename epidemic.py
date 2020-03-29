import csv
import numpy as np
import re
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.sparse import lil_matrix, csr_matrix


def load_maps():
    pop_map = np.zeros([8192, 8192]);
    de_map = np.zeros([8192, 8192]);
    with open('GEOSTAT_grid_POP_1K_2011_V2_0_1.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        next(spamreader)
        for row in spamreader:
            _, N, E = re.split('N|E', row[1])
            pop_map[int(N), int(E)] = float(row[0]);
            de_map[int(N), int(E)] = row[-2]=='DE';
    return pop_map, de_map

down = 16
pop_map, de_map = load_maps()
pop_map = pop_map.reshape(pop_map.shape[0]//down, down, pop_map.shape[1]//down, down).sum(-1).sum(1)
de_map = de_map.reshape(de_map.shape[0]//down, down, de_map.shape[1]//down, down).sum(-1).sum(1) > 0

N_nz, E_nz = np.where(pop_map)
i_to_k = np.where(pop_map.flat)[0]
m = i_to_k.size
k_to_i = np.full_like(pop_map.flat, np.nan)
k_to_i[i_to_k] = np.arange(m)
    
nnz = np.size(N_nz)

def compute_rij(radius):
    rij = lil_matrix((m, m))
    Ns_rel = np.arange(-radius, radius + 1)
    Es_rel = np.arange(-radius, radius + 1)
    ij = []
    print(i_to_k.size)
    for index in range(m):
        if index%1000 == 0:
            print(index)
        NEs = np.stack(np.meshgrid(N_nz[index] + Ns_rel, E_nz[index] + Es_rel)).reshape(2,-1)
        j_as_ks = np.ravel_multi_index(NEs, pop_map.shape)
        js_with_nan = k_to_i[j_as_ks]
        js = js_with_nan[~np.isnan(js_with_nan)]
        rij[index,js] = 1.0/len(js)
    return csr_matrix(rij)

rij0 = compute_rij(0)
rijd = compute_rij(1)

N = pop_map.flat[i_to_k]
tot_pop = np.sum(N)

def plot(x, y, z, xs, ys, za, tot_ts, axs):
    y_map = np.zeros_like(pop_map)
    y_map.flat[i_to_k] = y
    axs[0,0].clear()
    axs[0,1].clear()
    axs[1,0].clear()
    axs[1,1].clear()
    #plt.imshow(np.log(pop_map), origin='lowerleft')
    axs[0,0].imshow(np.log10((y_map+1e-7)/(pop_map+1e-7)), cmap='hot', origin='lowerleft')
    axs[0,1].imshow(y_map, cmap='hot', origin='lowerleft', clim=[0, 500000])
    axs[1,0].plot([0*np.sum(x) for x in xs])
    axs[1,0].plot([np.sum(y) for y in ys])
    axs[1,0].plot([np.sum(z) for z in zs])
    axs[1,1].semilogy([np.sum(x) for x in xs])
    axs[1,1].semilogy([np.sum(y) for y in ys])
    axs[1,1].semilogy([np.sum(z) for z in zs])
    axs[1,1].semilogy(tot_ts)
    plt.pause(0.02)

fig, axs = plt.subplots(2,2)

# Expected duration of disease in days
lam = 20

# Rates
r0_nominal = 0.1
rd_nominal = 0.02

# Time step
h = 1

# Number of daily travellers. Decrease to 2% after 200 days
def travellers(t):
    return 100000 if t < 200 else 2000

# Where does the disease start?
i0 = 4160

# x - # not yet infected people
# y - # currently infected
# z - # immune

x = N
y = np.zeros(m)
y[i0] = 1
z = np.zeros(m)

xnp1 = x
ynp1 = y
znp1 = z
tnp1 = 0
xs = []
ys = []
zs = []
tot_ts = []

for n in range(int(np.ceil(365*5/h))):
    print(n)

    t = h*n
    
    xn = xnp1
    yn = ynp1
    zn = znp1
    tn = tnp1
    Nn = xn+yn+zn

    xs.append(xn)
    ys.append(yn)
    zs.append(zn)
    tot_ts.append(tn)
    
    print(np.sum(xn), np.sum(yn), np.sum(zn))

    if n%100 == 0:
        plot(xn, yn, zn, xs, ys, zs, tot_ts, axs)

    # Add seasonal variation to rates
    seasonal_factor = (3 + np.cos(2*np.pi*t/365))/4
    r0 = r0_nominal*seasonal_factor
    rd = rd_nominal*seasonal_factor

    # Germany factor
    r0_vector = r0*(1 - 0.2*de_map.flat[i_to_k])
    rd_vector = rd*(1 - 0.2*de_map.flat[i_to_k])
    
    # Compute reaction matrix
    R = sparse.diags(r0_vector)*rij0 + sparse.diags(rd_vector)*rijd

    # Compute transitions
    hRyn = h*(R*(yn/Nn))
    x_to_y = xn - xn/(1+hRyn)
    y_to_z = h/lam*yn

    # Apply transitions
    xnp1 = xn - x_to_y
    ynp1 = yn + x_to_y - y_to_z
    znp1 = zn + y_to_z

    # Model far-distance travelling
    tot_y = np.sum(ynp1)
    t_y = (h*travellers(t)/tot_pop)*tot_y
    tp_y, tn0_y = np.modf(t_y*Nn/tot_pop)
    tn1_y = 1.0*(np.random.rand(np.size(tp_y)) < tp_y)
    tnp1_y = tn0_y + tn1_y
    tnp1 = np.sum(tnp1_y)
    print('travelling = {}'.format(tnp1))
    
    ynp1 = ynp1 - np.sum(tnp1_y)*ynp1/tot_y
    ynp1 = ynp1 + tnp1_y
    
plt.show()
