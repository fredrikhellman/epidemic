import numpy as np

def simulate(y_0, R0, REs, t_low, t_normal, t_max):
    h = 0.01
    
    REs = np.array(REs)
    
    r_low = REs
    r_normal = np.full_like(REs, R0)

    r_of_t = lambda t: (r_normal if t < t_low else
                        r_low if t < t_normal else
                        r_normal)

    # x - # not yet infected people
    # y - # currently infected
    # z - # immune

    xs = [np.full_like(REs, 1 - y_0)]
    ys = [np.full_like(REs, y_0)]
    zs = [np.full_like(REs, 0)]
    ts = [0]

    for n in range(int(np.ceil(t_max/h))):
        t = h*n

        # Compute transitions
        hRy = h*r_of_t(t)*ys[-1]
        x_to_y = xs[-1] - xs[-1]/(1+hRy)
        y_to_z = h*ys[-1]

        # Apply transitions
        xs.append(xs[-1] - x_to_y)
        ys.append(ys[-1] + x_to_y - y_to_z)
        zs.append(zs[-1] + y_to_z)
        ts.append(t)

    return ts, xs, ys, zs
