import numpy as np

# Demodulation (array input)
def AC(imgs: list):
    return (2 ** 0.5 / 3) * (((imgs[0] - imgs[1]) ** 2 + (imgs[1] - imgs[2]) ** 2 + (imgs[2] - imgs[0]) ** 2) ** 0.5)

def DC(imgs: list):
    return sum(imgs) / len(imgs)

def mu_eff(mu_a, mu_tr, f):
    a = (2 * np.pi * f) ** 2
    return mu_tr * (3 * (mu_a / mu_tr) + a / mu_tr ** 2) ** 0.5

def ac_diffuse(refr_index):
    return 0.0636 * refr_index + 0.668 + 0.710 / refr_index - 1.44 / (refr_index ** 2)

def diffusion_approximation(n, mu_a, mu_sp, f_ac):
    # AC diffuse reflectance from Diffusion Approximation
    R_eff = 0.0636 * n + 0.668 + 0.710 / n - 1.44 / (n ** 2)
    A = (1 - R_eff) / (2 * (1 + R_eff))
    mu_tr = mu_a + mu_sp
    ap = mu_sp / mu_tr

    r_ac = (3 * A * ap) / (((2 * np.pi * f_ac) / mu_tr) ** 2 + ((2 * np.pi * f_ac) / mu_tr) * (1 + 3 * A) + 3 * A)
    r_dc = (3 * A * ap) / (3 * (1 - ap) + (1 + 3 * A) * np.sqrt(3 * (1 - ap)) + 3 * A)

    return r_ac, r_dc