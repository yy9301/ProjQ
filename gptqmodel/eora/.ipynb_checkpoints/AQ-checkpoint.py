# loftq initial Adapter 得到Q，然后通过闭解得到A

from typing import Sequence, Tuple

import torch
from torch import Tensor

from ..utils.logger import setup_logger
from ..utils.rocm import IS_ROCM

log = setup_logger()

def eora_process_input(
        input: Tensor,
        name: str,
        sample_size: int,
        device: torch.device,
) -> Tuple[int, torch.Tensor, float]:
    """Prepare the per-batch covariance contribution required for EoRA.

    The contribution remains on the originating device so multi-GPU execution
    can accumulate locally before a single merge step.
    """

    inp = input[0].to(device=device, dtype=torch.float32)
    if inp.dim() == 2:
        inp = inp.unsqueeze(0)

    batch = inp.shape[0]
    adds = torch.matmul(inp.transpose(1, 2), inp)
    adds_sum = torch.sum(adds, dim=0).detach()

    contribution = adds_sum.to(dtype=torch.float32)
    contribution /= float(sample_size)

    # Adding batch to denominator is only for mathematical stability
    scale = float(sample_size) / (float(sample_size) + float(batch))

    del inp, adds, adds_sum

    return batch, contribution, scale


def merge_eora_segments(segments: Sequence[Tuple[torch.Tensor, float]]) -> torch.Tensor:
    """Combine pre-aggregated EoRA segments using their scale products.

    Each segment entry is a tuple ``(total, scale_product)`` where ``total`` is
    the sequential accumulation result for that segment starting from zero, and
    ``scale_product`` is the product of per-batch scale factors encountered in
    the same segment.  The function mutates the first segment tensor in place
    and returns it as the merged result.
    """
    if not segments:
        raise ValueError("EoRA merge requires at least one segment.")

    result: torch.Tensor | None = None
    for total, scale_product in segments:
        if result is None:
            result = total
        else:
            result.mul_(float(scale_product))
            result.add_(total)

    assert result is not None
    return result
def debug_log_tensor(name, T, logger=log, max_elems=8):
    try:
        if T is None:
            logger.info(f"{name}: None")
            return
        s = f"{name}: shape={tuple(T.shape)}, dtype={T.dtype}, device={T.device}"
        # promote to float32 for stable stats if needed
        T_float = T.detach().to(dtype=torch.float32)
        finite_mask = torch.isfinite(T_float)
        any_nan = torch.isnan(T_float).any().item()
        any_inf = (~finite_mask).any().item() and (torch.isinf(T_float).any().item() if torch.isfinite(T_float).all() is False else (~finite_mask).any().item())
        s += f", any_nan={any_nan}, any_nonfinite={any_inf}"
        try:
            s += f", min={float(torch.min(T_float).item()):.6g}, max={float(torch.max(T_float).item()):.6g}, mean={float(torch.mean(T_float).item()):.6g}, norm={float(torch.norm(T_float).item()):.6g}"
        except Exception:
            pass
        # show head elements
        flat = T_float.flatten()
        head = flat[:max_elems].cpu().numpy().tolist()
        s += f", head={head}"
        logger.info(s)
    except Exception as e:
        logger.info(f"{name}: debug_log_tensor error: {e}")
        
def randomized_svd(A, rank, oversample=8, n_iter=2):
    orig_dtype = A.dtype
    compute_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    A = A.to(dtype=compute_dtype)

    m, n = A.shape
    r = min(rank + oversample, min(m, n))  

    Omega = torch.randn(n, r, device=A.device, dtype=compute_dtype)
    Omega = Omega / torch.sqrt(torch.tensor(r, dtype=compute_dtype, device=A.device))  # 标准化方差


    Y = A @ Omega  # (m, r)
    for _ in range(max(0, n_iter)):
        Y = A @ (A.T @ Y)
        Y = Y / (torch.norm(Y, dim=0, keepdim=True) + 1e-12)

    Q, _ = torch.linalg.qr(Y, mode="reduced")

    B = Q.T @ A  # (r, n)
    if n > r:
        Ub, S, Vh = torch.linalg.svd(B.T, full_matrices=False)
        Vh, Ub = Ub.T, Vh.T
    else:
        Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)

    U = Q @ Ub  # (m, r)

    U = U[:, :rank].to(orig_dtype)
    S = S[:rank].to(orig_dtype)
    Vh = Vh[:rank, :].to(orig_dtype)

    return U, S, Vh

def randomized_psd_sqrt(C, k, n_iter=2):
    """
    Approximate Y such that C ≈ Y Y^T
    C must be symmetric PSD
    """
    n = C.shape[0]
    dtype = C.dtype
    device = C.device

    # Random probe
    Omega = torch.randn(n, k, device=device, dtype=dtype)
    Y0 = C @ Omega

    for _ in range(n_iter):
        Y0 = C @ (C @ Y0)

    # Orthonormal basis
    Q, _ = torch.linalg.qr(Y0, mode="reduced")

    # Small projected matrix
    B = Q.T @ C @ Q  # (k, k)

    # Exact eigendecomposition on small matrix
    eigvals, eigvecs = torch.linalg.eigh(B)
    eigvals = torch.clamp(eigvals, min=1e-6)

    # Construct Y
    Y = Q @ eigvecs @ torch.diag(torch.sqrt(eigvals))
    return Y

def eora_compute_lora(
        w_wq_delta: Tensor, # need the w (original weight) and wq (quantized qweight) delta in float32
        X: Tensor,
        name: str,
        eigen_scaling_diag_matrix: torch.Tensor, # 合并后的协方差矩阵（来自merge_eora_segments）
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
) -> Tuple[Tensor, Tensor]:

    assert w_wq_delta.dtype == torch.float32

    # save this later for SVD--协方差矩阵转换为float64（提高特征值分解精度）
    raw_scaling_diag_matrix = eigen_scaling_diag_matrix.to(device=device, dtype=torch.float64)
    X = X.squeeze(0).transpose(0, 1).to(dtype=torch.float32) # (4096, 2048)
    if IS_ROCM:
        # hip cannot resolve linalg ops
        original_backend = torch.backends.cuda.preferred_linalg_library()
        torch.backends.cuda.preferred_linalg_library(backend="magma")
    debug_log_tensor(f"w_wq_delta", w_wq_delta)
    debug_log_tensor(f"activation", X)
    R = w_wq_delta @ X # (4096, 4096)x(4096, 2048)
    C = R @ X.T
    truc_u, truc_s, truc_v = randomized_svd(A=C,rank=rank,oversample=rank+5,n_iter=2)

    # lowrank_r = rank
    # truc_s = S[:lowrank_r]
    # truc_u = U[:, :lowrank_r]
    # truc_v = V[:lowrank_r, :]
    truc_sigma = torch.diag(torch.sqrt(truc_s))
    X_pinv = torch.linalg.pinv(X)
    # B = torch.matmul(truc_u, truc_sigma).to(dtype=dtype) # default to float16, check if we should save to float32
    # A = torch.matmul(truc_sigma, truc_v).to(dtype=torch.float32) # default to float16, check if we should save to float32，这里的A是adapter的右边，也就是我们的Q (16, 2048)
    A = truc_v #(16, 4096)
    debug_log_tensor(f"A", A)
    # A_X = A @ X # [16, 2048]
    # inv = torch.inverse(A_X @ A_X.T) # [16,16]
    AAT = A @ A.T
    inv = torch.linalg.inv(AAT + 1e-6 * torch.eye(rank, device=A.device)) # [16,16]
    debug_log_tensor(f"inv", inv) 
    B = - w_wq_delta @ A.T @ inv #这里的B相当于我们需要的A
    
    

    # del Y, U, S, V,
    del w_wq_delta, raw_scaling_diag_matrix
    del truc_s, truc_u, truc_v, truc_sigma

    # revert linalg backend
    if IS_ROCM:
        torch.backends.cuda.preferred_linalg_library(original_backend)

    return A, B

# def eora_compute_lora(
#         w_wq_delta: Tensor, # need the w (original weight) and wq (quantized qweight) delta in float32
#         name: str,
#         eigen_scaling_diag_matrix: torch.Tensor,
#         rank: int,
#         dtype: torch.dtype,
#         device: torch.device,
# ) -> Tuple[Tensor, Tensor]:

#     assert w_wq_delta.dtype == torch.float32

#     # save this later for SVD
#     raw_scaling_diag_matrix = eigen_scaling_diag_matrix.to(device=device, dtype=torch.float64)

#     if IS_ROCM:
#         # hip cannot resolve linalg ops
#         original_backend = torch.backends.cuda.preferred_linalg_library()
#         torch.backends.cuda.preferred_linalg_library(backend="magma")

#     L, Q = torch.linalg.eigh(raw_scaling_diag_matrix)

#     if (L < 0).any():
#         ## When expanding the calibration data size for EoRA, I suggest maintaining the balance by allocating 50% to general input (C4) and the remaining 50% to downstream task data.
#         log.warn(f"Found negative eigenvalues in `{name}`. Please increase your calibration data set for EoRA.")
#         minimum = torch.min(L[L > 0])
#         L[L < 0] = minimum

#     sqrtEigenvalues = torch.sqrt(L)
#     scaling_diag_matrix = Q @ torch.diag(sqrtEigenvalues)

#     scaling_matrix_inv = torch.diag(1/sqrtEigenvalues) @ Q.T

#     scaling_diag_matrix = scaling_diag_matrix.to(dtype=torch.float32)
#     scaling_matrix_inv = scaling_matrix_inv.to(dtype=torch.float32)

#     delta_scale = torch.matmul(w_wq_delta, scaling_diag_matrix)

#     U, S, V = torch.linalg.svd(delta_scale, full_matrices=False)
#     lowrank_r = rank
#     truc_s = S[:lowrank_r]
#     truc_u = U[:, :lowrank_r]
#     truc_v = torch.matmul(V[:lowrank_r, :], scaling_matrix_inv)
#     truc_sigma = torch.diag(truc_s)

#     sqrtS = torch.sqrt(truc_sigma)
#     B = torch.matmul(truc_u, sqrtS).to(dtype=dtype) # default to float16, check if we should save to float32
#     A = torch.matmul(sqrtS, truc_v).to(dtype=dtype) # default to float16, check if we should save to float32


#     del L, Q, U, S, V,
#     del w_wq_delta, raw_scaling_diag_matrix, sqrtEigenvalues, scaling_diag_matrix, scaling_matrix_inv, delta_scale
#     del truc_s, truc_u, truc_v, truc_sigma, sqrtS

#     # revert linalg backend
#     if IS_ROCM:
#         torch.backends.cuda.preferred_linalg_library(original_backend)

#     return A, B
