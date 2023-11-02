import torch


def pointwise_ABCd(A, B, C, d=None):
    """Prepare for 3rd order cumulant computation
    Args:
        A (2D tensor)
        B (2D tensor)
        C (2D tensor)
        d (1D tensor, optional): _description_. Defaults to None.
    Returns:
        M = M[ijk] -- 3D tensor
        M^ijk = \sum_m^d (A^im B^jm C^km)*d^m
    """
    A, B, C = A.float(), B.float(), C.float()
    Id = torch.eye(B.size(1))
    if d is None:
        d = Id
    else:
        d = torch.diag(d)
    d = d.float()
    # diagnalize B
    tmp_B = B.unsqueeze(2).expand(*B.size(), B.size(1))
    DiagB = tmp_B * Id
    DiagB = torch.matmul(DiagB.float(), d)

    BC = torch.matmul(DiagB, torch.t(C))

    BC_A = torch.transpose(torch.matmul(torch.transpose(BC, 1, 2), torch.t(A)), 1, 2)

    return torch.transpose(BC_A, 0, 1)


def compute_cumulants(R, L, exo_baseline, cov_obs=None, return_full_K=False):
    """Compute theoretical cumulants C and K from L and R

    Args:
        - R (tensor, 2d matrix): dim x dim, numpy array or torch tensor
        - L (tensor, 1d vector): dim, numpy array or torch tensor
        - exo_baseline (tensor, 1d vector): dim//2 , numpy array or torch tensor.If exo_baseline=0, classical hawkes, otherwise, hawkes with shot noise

        - cov_obs (2d matrix, optional): dim x dim. Defaults to None. If not None, using this observable covariance matrix to compute K.
        - return_full_K (bool, optional): return full K (K_ijk --> 3D tensor) or not (only K_ijj --> 2D tensor). Defaults to False.

    Returns: (torch tensors)
        if return_full_K:
            C (2d matrix), K (3d matrix): C[ij], K[ijk]
        else:
            C (2d matrix), K (2d matrix): C[ij], K[ijj]
    """
    R = torch.Tensor(R)
    L = torch.Tensor(L)
    exo_baseline = torch.Tensor(exo_baseline)
    # ----------------------------------------------------
    dim = R.size(0)
    ### prepare
    Lx = torch.cat((exo_baseline, exo_baseline), axis=0)
    diag_Lx = torch.diag(Lx)
    diag_L = torch.diag(L)
    eyes = torch.eye(dim // 2)
    zeros = torch.zeros(dim // 2, dim // 2)

    cross_diag_eyes = torch.cat(
        (torch.cat((zeros, eyes), axis=1), torch.cat((eyes, zeros), axis=1)), axis=0
    )
    cross_diag_Lx = torch.mm(cross_diag_eyes, diag_Lx)
    cov_x = diag_Lx + cross_diag_Lx

    # observable covariance = R * Sigma * R^T
    C = torch.mm(torch.mm(R, diag_L + cross_diag_Lx), torch.t(R))

    # skewness
    if cov_obs is None:
        cov_obs = C
    cross_cov_mtx = torch.mm(R, cov_x) - cov_x
    Cov_all = torch.cat(
        (
            torch.cat(
                (cov_obs - cross_cov_mtx - cross_cov_mtx.T - cov_x, cross_cov_mtx),
                axis=1,
            ),
            torch.cat((cross_cov_mtx.T, cov_x), axis=1),
        ),
        axis=0,
    )
    ## expand R to R_all
    R_x = torch.cat(
        (torch.cat((eyes, zeros), axis=1), torch.cat((eyes, zeros), axis=1)), axis=0
    )
    R_cross = torch.mm(R, R_x) - R_x
    R_all = torch.cat(
        (
            torch.cat((R, R_cross), axis=1),
            torch.cat((torch.zeros_like(R), R_x), axis=1),
        ),
        axis=0,
    )
    RRC = pointwise_ABCd(R_all, R_all, Cov_all)
    L_all = torch.cat((L - Lx, Lx), axis=0)
    K_all = (
        RRC
        + torch.transpose(RRC, 1, 2)
        + torch.transpose(RRC, 0, 2)
        - 2 * pointwise_ABCd(R_all, R_all, R_all, L_all)
    )
    ## compute observable K :
    ## In case of dim4, K_all 8 x 8 x 8 --> K 4 x 4 x 4
    ## For ex. K[112] = K_all[112] + K_all[116] + K_all[152] + K_all[156] +
    ##                  K_all[512] + K_all[516] + K_all[552] + K_all[556]
    eyes_2 = torch.cat((torch.eye(R.size(0)), torch.eye(R.size(0))), axis=1)
    K = torch.matmul(torch.matmul(eyes_2, K_all), torch.t(eyes_2))
    K = torch.transpose(torch.matmul(eyes_2, torch.transpose(K, 0, 1)), 0, 1)

    if return_full_K:  # return K[ijk]
        return C, K
    else:  # return only K[ijj]
        K = torch.diagonal(K, dim1=0, dim2=1)
        return C, K


# def compute_compressed_cumulants(C, K):
#     """_summary_

#     Args:
#         C (tensor): 4 x 4
#         K (tensor): 4 x 4 x 4
#     """
#     M = torch.Tensor([[1, -1, 0, 0], [0, 0, 1, -1]])
#     C_2 = torch.matmul(torch.matmul(M, C), torch.transpose(M, 0, 1))
#     K_2 = torch.matmul(torch.matmul(M, K), torch.transpose(M, 0, 1))
#     K_2 = torch.transpose(torch.matmul(M, torch.transpose(K_2, 0, 1)), 0, 1)
#     return C_2, K_2
