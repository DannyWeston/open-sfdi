import numpy as np

def unwrap_phase(wrapped):
    if wrapped.ndim == 1:
        return __unwrap_phase_1d(wrapped)
    
    if wrapped.ndim == 2:
        return __unwrap_phase_2d(wrapped)
    
    raise Exception("Only one and two-dimensional unwraps are supported!")

def __unwrap_phase_1d(wrapped):
    # Accumulate wraps of pi and store in k
    k = 0

    addon = np.zeros(wrapped.shape)

    for i in range(0, len(wrapped) - 1):
        difference = wrapped[i + 1] - wrapped[i]
        if np.pi < difference:
            k -= (2.0 * np.pi)

        elif difference < -np.pi:
            k += (2.0 * np.pi)

        addon[i] = k

    return wrapped + addon

def __unwrap_phase_2d(wrapped):
    # TODO: Maybe possible to do this in place?

    a, b = wrapped.shape

    unwrapped = np.zeros((a, b))
    for i in range(0, a):
        unwrapped[i, :] = __unwrap_phase_1d(wrapped[i, :])

    for j in range(0, b):
        unwrapped[:, j] = __unwrap_phase_1d(wrapped[:, j])

    return unwrapped