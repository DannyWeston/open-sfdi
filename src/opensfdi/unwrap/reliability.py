import numpy as np

from itertools import zip_longest

# Fast unwrapping 2D phase image using the algorithm given in:
#     M. A. Herráez, D. R. Burton, M. J. Lalor, and M. A. Gdeisat,
#     "Fast two-dimensional phase-unwrapping algorithm based on sorting by
#     reliability following a noncontinuous path", Applied Optics, Vol. 41,
#     Issue 35, pp. 7437-7444 (2002).
#
# If using this code for publication, please kindly cite the following:
# * M. A. Herraez, D. R. Burton, M. J. Lalor, and M. A. Gdeisat, "Fast
#   two-dimensional phase-unwrapping algorithm based on sorting by reliability
#   following a noncontinuous path", Applied Optics, Vol. 41, Issue 35,
#   pp. 7437-7444 (2002).
# * M. F. Kasim, "Fast 2D phase unwrapping implementation in MATLAB",
#   https://github.com/mfkasim91/unwrap_phase/ (2017).

# Input:
# * img: The wrapped phase image either from -pi to pi or from 0 to 2*pi.
#        If there are unwanted regions, it should be filled with NaNs.

# Output:
# * res_img: The unwrapped phase with arbitrary offset.
#
# Author:
#     Muhammad F. Kasim, University of Oxford (2017)
#     Email: firman.kasim@gmail.com

def unwrap_phase(wrapped, relationship=None):
    if not relationship:
        relationship = __default_rel

    if wrapped.ndim == 2:
        return __unwrap_phase_2d(wrapped, relationship)
    
    raise Exception("Only 2D phase unwrapping is supported by this algorithm!")

def __default_rel(x):
    with np.errstate(divide='ignore'):
        return 1.0 / x

def __unwrap_phase_2d(wrapped, relationship):
    w_x, w_y = wrapped.shape

    # Get the reliability
    reliability = __get_reliability_2d(wrapped, relationship)

    # Get the edges
    hori, vert = __get_edges_2d(reliability)

    # Combine all edges and sort
    edges = np.append(hori, vert)

    np.set_printoptions(precision=1)

    print(edges, end='\n\n')

    num_edges = w_x * w_y

    # Sort by highest reliability - collect indices instead of values
    sorted = np.flip(np.argsort(edges))

    print(sorted, end='\n\n')

    # Get the indices of pixels adjacent to the edges
    neighbour1 = np.mod(sorted - 1, num_edges) + 1
    neighbour2 = neighbour1 + 1 + (w_y - 1) * (sorted > num_edges)


    # TODO: Fix function



    groups = np.reshape(np.arange(0, wrapped.size, 1, dtype=int), num_edges)

    # label the group
    is_grouped = np.zeros_like(groups)
    group_members = np.empty_like(groups)

    for i in range(is_grouped.size):
        group_members[i] = i
    
    num_members_group = np.ones(edges_size)

    # propagate the unwrapping
    res_img = wrapped
    num_nan = sum(np.isnan(edges))

    # for i in range(num_nan+1, len(edge_sort_idx)):
    #     # get the indices of the adjacent pixels
    #     idx1 = idxs1(i)
    #     idx2 = idxs2(i)
    
    #     # skip if they belong to the same group
    #     if (group(idx1) == group(idx2)):
    #         continue
    
    #     # idx1 should be ungrouped (swap if idx2 ungrouped and idx1 grouped)
    #     # otherwise, activate the flag all_grouped.
    #     # The group in idx1 must be smaller than in idx2. If initially
    #     # group(idx1) is larger than group(idx2), then swap it.
    
    #     all_grouped = 0
    #     if is_grouped(idx1):
    #         if not is_grouped(idx2):
    #             idxt = idx1
    #             idx1 = idx2
    #             idx2 = idxt
    #         elif num_members_group(group(idx1)) > num_members_group(group(idx2)):
    #             idxt = idx1
    #             idx1 = idx2
    #             idx2 = idxt
    #             all_grouped = 1
    #         else:
    #             all_grouped = 1
            
    #     # calculate how much we should add to the idx1 and group
    #     dval = floor((res_img(idx2) - res_img(idx1) + np.pi) / (2.0 * np.pi)) * 2 * np.pi
    
    #     # which pixel should be changed
    #     g1 = group(idx1)
    #     g2 = group(idx2)
    
    #     if all_grouped: 
    #         pix_idxs = group_members{g1}

    #     else: 
    #         pix_idxs = idx1
    
    #     # add the pixel value
    #     if dval != 0: 
    #         res_img[pix_idxs] = res_img[pix_idxs] + dval
    
    #     # change the group
    #     len_g1 = num_members_group(g1)
    #     len_g2 = num_members_group(g2)
    #     group_members[g2][len_g2 + 1 : len_g2 + len_g1] = pix_idxs
    #     group[pix_idxs] = g2 # assign the pixels to the new group
    #     num_members_group[g2] = num_members_group(g2) + len_g1
    
    #     # mark idx1 and idx2 as already being grouped
    #     is_grouped[idx1] = 1
    #     is_grouped[idx2] = 1

def __get_reliability_2d(img, relationship=None):

    # Diagonals
    img_in1_jn1 = img[:-2, 2:]      # i = -1, j = -1
    img_in1_jp1 = img[:-2, :-2]     # i = -1, j = +1
    img_ip1_jp1 = img[2:, :-2]      # i = +1, j = +1
    img_ip1_jn1 = img[2:, 2:]       # i = +1, j = -1

    # Orthogonal
    img_i_jp1   = img[1:-1, :-2]    # i = 0, j = +1 
    img_i_jn1   = img[1:-1, 2:]     # i = 0, j = -1
    img_ip1_j   = img[2:, 1:-1]     # i = +1, j = 0
    img_in1_j   = img[:-2, 1:-1]    # i = -1, j = 0

    # Central
    img_i_j     = img[1:-1, 1:-1]   # i = 0, j = 0

    # Determine positive or negative modulus pi
    gamma_mod = lambda x: np.sign(x) * np.mod(np.abs(x), np.pi)

    # H = gamma( Phi_I(-1, 0) - Phi_I(0, 0) ) - gamma( Phi_I(0, 0) - Phi_I(1, 0) )
    H  = gamma_mod(img_in1_j - img_i_j) - gamma_mod(img_i_j - img_ip1_j)

    # V = gamma( Phi_I(0, -1) - Phi_I(0, 0) ) - gamma( Phi_I(0, 0) - Phi_I(0, 1) )
    V  = gamma_mod(img_i_jn1   - img_i_j) - gamma_mod(img_i_j - img_i_jp1)

    # D1 = gamma( Phi_I(-1, -1) - Phi_I(0, 0) ) - gamma( Phi_I(0, 0) - Phi_I(1, 1) )
    D1 = gamma_mod(img_in1_jn1 - img_i_j) - gamma_mod(img_i_j - img_ip1_jp1)

    # D2 = gamma( Phi_I(-1, +1) - Phi_I(0, 0) ) - gamma( Phi_I(0, 0) - Phi_I(1, -1) )
    D2 = gamma_mod(img_in1_jp1 - img_i_j) - gamma_mod(img_i_j - img_ip1_jn1)

    # D = sqrt(H^2 + V^2 + D1^2 D2^2)
    D = np.sqrt(H * H + V * V + D1 * D1 + D2 * D2)

    # Calculate a reliability score for non-border pixels
    rel = np.zeros_like(img)
    rel[1:-1, 1:-1] = relationship(D)

    # Any NaNs in the non-border pixels reduce to 0
    # TODO: Maybe consider putting this in the relationship function?
    rel[np.isnan(rel)] = 0  # Set to NaNs to be unreliable

    return rel

def __print_edges(hori, vert):
    hori2 = hori.tolist()
    vert2 = vert.tolist()

    rows = zip_longest(vert2, hori2, fillvalue=[])

    for (v, h) in rows:
        for x in v: print(f'\t{x:.2f}', end='\t')
        print()

        for x in h: print(f'{x:.2f}\t', end='\t')
        print()

def __get_edges_2d(rel):
    # Image could be in a format such that width != height
    # Therefore, need to pad edges with necessary nans so the
    # the shape of vert matches hori
    #
    # This should be accounted for wherever the edges are used
    x, y = rel.shape

    # [rel(2:end, 1:end) + rel(1:end-1, 1:end); nan(1, Nx)];
    hori = np.array(rel[:, 1:] + rel[:, :-1])
    vert = np.array(rel[1:, :] + rel[:-1, :])

    # These edges need to be padded before usage
    hori = np.hstack((hori, np.full((x, 1), -1))).flatten()
    vert = np.vstack((vert, np.full((1, y), -1))).flatten()

    return hori, vert
