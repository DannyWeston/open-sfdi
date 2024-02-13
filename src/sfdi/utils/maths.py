import numpy as np

# Demodulation (array input)
def AC(imgs: list):
    return (2 ** 0.5 / 3) * (((imgs[0] - imgs[1]) ** 2 + (imgs[1] - imgs[2]) ** 2 + (imgs[2] - imgs[0]) ** 2) ** 0.5)

def DC(imgs: list):
    return sum(imgs) / len(imgs)

def mu_eff(mu_a, mu_tr, f):
    a = (2 * np.pi * f) ** 2
    return mu_tr * (3 * (mu_a / mu_tr) + a / mu_tr ** 2) ** 0.5

def diffusion_approximation(A, ap, mu_tr, f_ac):
    r_ac = (3 * A * ap) / (((2 * np.pi * f_ac) / mu_tr) ** 2 + ((2 * np.pi * f_ac) / mu_tr) * (1 + 3 * A) + 3 * A)
    r_dc = (3 * A * ap) / (3 * (1 - ap) + (1 + 3 * A) * np.sqrt(3 * (1 - ap)) + 3 * A)

    return r_ac, r_dc