import numpy as np

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
        relationship = lambda x: 1 / x

    if wrapped.ndim == 2:
        return __unwrap_phase_2d(wrapped, relationship)
    
    raise Exception("Only 2D phase unwrapping is supported by this algorithm!")

def __unwrap_phase_2d(wrapped, relationship):
    w_x, w_y = wrapped.shape

    # Get the reliability
    reliability = __get_reliability_2d(wrapped, relationship)

    print()
    print(reliability)

    # Get the edges
    vert_edges, hori_edges = __get_edges_2d(reliability)

    print()
    print(vert_edges)

    print()
    print(hori_edges)

    return reliability

    # # Combine all edges and sort it
    # edges = [hori_edges, vert_edges]
    # edges_size = w_x * w_y

    # # Sort into descending order
    # edges_sort = edges[::-1].sort()

    # # get the indices of pixels adjacent to the edges
    # idxs1 = np.mod(edge_sort_idx - 1, edge_bound_idx)

    # group = np.reshape(Ny * Nx, 1)

    # p = [1 : numel[wrapped]]

    # idxs2 = idxs1 + 1 + (Ny - 1) * (edge_sort_idx <= edge_bound_idx)

    

    # # label the group
    # is_grouped = np.zeros(Ny*Nx,1)
    # group_members = [Ny*Nx,1]

    # for i in range(len(is_grouped)):
    #     group_members[i] = i
    
    # num_members_group = np.ones(Ny * Nx, 1)

    # # propagate the unwrapping
    # res_img = wrapped
    # num_nan = sum(np.isnan(edges))

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

def __get_reliability_2d(img, relationship):

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

    rel = np.empty_like(img)
    rel[:] = np.nan # Fill with NaNs for now

    # Set all non-border pixels to their relability score
    rel[1:-1, 1:-1] = relationship(D)

    # Any NaNs in the non-border pixels reduce to 0 (really small phase change)
    # TODO: Maybe consider putting this in the relationship function?
    rel[1:-1, 1:-1][np.isnan(rel[1:-1, 1:-1])] = 0 # No phase change?

    return rel

def __get_edges_2d(rel):
    x, y = rel.shape
    vert = [rel[:, :-1] + rel[:, 1:], np.full((1, x), np.nan)]
    hori = [rel[:-1, :] + rel[1:, :], np.full((y, 1), np.nan)]

    return vert, hori
