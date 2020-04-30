import numpy as np

def simulate(lam, R0, REs, t_low, t_normal, t_max, pop):
    REs = np.array(REs)

    m = len(REs)

    # Infection rates at time t (days)
    r_low = np.array(REs)/lam
    r_normal = m*[R0/lam]

    rlist_of_t = lambda t: (r_normal if t < t_low else
                            r_low if t < t_normal else
                            r_normal)

    # Population sizes
    N = np.array(m*[pop])

    r_of_t = lambda t: np.array(rlist_of_t(t))

    # Time step
    h = 0.1

    # x - # not yet infected people
    # y - # currently infected
    # z - # immune

    x = N - 1
    y = np.ones_like(x)
    z = np.zeros_like(x)

    xs = [x]
    ys = [y]
    zs = [z]
    ts = [0]

    for n in range(int(np.ceil(t_max/h))):
        t = h*n

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

    return ts, xs, ys, zs
