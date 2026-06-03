import torch
from typing import Sequence, Tuple

from ..utils.logger import setup_logger
from ..utils.rocm import IS_ROCM

log = setup_logger()


def randomized_svd(A, rank, oversample=8, n_iter=2):
    orig_dtype = A.dtype
    compute_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    A = A.to(dtype=compute_dtype)

    m, n = A.shape
    r = min(rank + oversample, min(m, n))

    Omega = torch.randn(n, r, device=A.device, dtype=compute_dtype)
    Omega = Omega / torch.sqrt(torch.tensor(r, dtype=compute_dtype, device=A.device))

    Y = A @ Omega
    for _ in range(max(0, n_iter)):
        Y = A @ (A.T @ Y)
        Y = Y / (torch.norm(Y, dim=0, keepdim=True) + 1e-12)

    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.T @ A

    if n > r:
        Ub, S, Vh = torch.linalg.svd(B.T, full_matrices=False)
        Vh, Ub = Ub.T, Vh.T
    else:
        Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)

    U = Q @ Ub
    U = U[:, :rank].to(orig_dtype)
    S = S[:rank].to(orig_dtype)
    Vh = Vh[:rank, :].to(orig_dtype)

    return U, S, Vh


def lordq_process_input(
        input: torch.Tensor,
        name: str,
        sample_size: int,
        device: torch.device,
) -> Tuple[int, torch.Tensor, float]:
    inp = input[0].to(device=device, dtype=torch.float32)
    if inp.dim() == 2:
        inp = inp.unsqueeze(0)

    batch = inp.shape[0]
    adds = torch.matmul(inp.transpose(1, 2), inp)
    adds_sum = torch.sum(adds, dim=0).detach()

    contribution = adds_sum.to(dtype=torch.float32)
    contribution /= float(sample_size)

    scale = float(sample_size) / (float(sample_size) + float(batch))

    del inp, adds, adds_sum
    return batch, contribution, scale


def merge_lordq_segments(segments: Sequence[Tuple[torch.Tensor, float]]) -> torch.Tensor:
    if not segments:
        raise ValueError("lordq merge requires at least one segment.")

    result: torch.Tensor | None = None
    for total, scale_product in segments:
        if result is None:
            result = total
        else:
            result.mul_(float(scale_product))
            result.add_(total)
    assert result is not None
    return result


def lordq_compute_lora(
        w_wq_delta: torch.Tensor,
        name: str,
        eigen_scaling_diag_matrix: torch.Tensor,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert w_wq_delta.dtype == torch.float32

    raw_scaling_diag_matrix = eigen_scaling_diag_matrix.to(device=device, dtype=torch.float64)

    if IS_ROCM:
        original_backend = torch.backends.cuda.preferred_linalg_library()
        torch.backends.cuda.preferred_linalg_library(backend="magma")

    Sx, Ux = torch.linalg.eigh(raw_scaling_diag_matrix)
    Sx_pos = torch.clamp(Sx, min=1e-6)
    Y = Ux @ torch.diag(torch.sqrt(Sx_pos))
    # Y = torch.linalg.cholesky(raw_scaling_diag_matrix)
    Y = Y.to(dtype=torch.float32)
    # Y_inv = torch.linalg.inv(Y)
    Y_inv = torch.diag(1.0 / torch.sqrt(Sx_pos)) @ Ux.T
    Y_inv = Y_inv.to(dtype=torch.float32)
    RY = w_wq_delta @ Y
    # RY = RY.to(dtype=torch.float32)
  
    jitter = 1e-6 * torch.randn_like(RY)
    truc_u, truc_s, truc_v = randomized_svd(A=RY+jitter,rank=rank,oversample=rank+5,n_iter=2)
    truc_sigma = torch.diag(truc_s)
    sqrtS = torch.sqrt(truc_sigma)
    B = truc_u @ sqrtS.to(dtype=dtype) # default to float16, check if we should save to float32
    A = (sqrtS @ truc_v @ Y_inv).to(dtype=dtype) # default to float16, check if we should save to float32

    del Y, w_wq_delta, raw_scaling_diag_matrix
    del truc_s, truc_u, truc_v, truc_sigma

    if IS_ROCM:
        torch.backends.cuda.preferred_linalg_library(original_backend)

    return A, B
