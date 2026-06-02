# lordq

from typing import Sequence, Tuple

import torch
from torch import Tensor

from ..utils.logger import setup_logger
from ..utils.rocm import IS_ROCM

log = setup_logger()

def lordq_process_input(
        input: Tensor,
        name: str,
        sample_size: int,
        device: torch.device,
) -> Tuple[int, torch.Tensor, float]:
    """Prepare the per-batch covariance contribution required for lordq.

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


def merge_lordq_segments(segments: Sequence[Tuple[torch.Tensor, float]]) -> torch.Tensor:
    """Combine pre-aggregated lordq segments using their scale products.

    Each segment entry is a tuple ``(total, scale_product)`` where ``total`` is
    the sequential accumulation result for that segment starting from zero, and
    ``scale_product`` is the product of per-batch scale factors encountered in
    the same segment.  The function mutates the first segment tensor in place
    and returns it as the merged result.
    """
    if not segments:
        raise ValueError("lordq merge requires at least one segment.")

    result: torch.Tensor | None = None
    for total, scale_product in segments:
        if result is None:
            result = total
        else:
            result.mul_(float(scale_product))
            result.add_(total)
    # result:输入激活值的总体协方差矩阵
    assert result is not None
    return result

def lordq_compute_lora(
        w_wq_delta: Tensor, # need the w (original weight) and wq (quantized qweight) delta in float32
        name: str,
        eigen_scaling_diag_matrix: torch.Tensor, # 合并后的协方差矩阵（来自merge_eora_segments）
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
) -> Tuple[Tensor, Tensor]:

    assert w_wq_delta.dtype == torch.float32

    # save this later for SVD
    raw_scaling_diag_matrix = eigen_scaling_diag_matrix.to(device=device, dtype=torch.float64)

    if IS_ROCM:
        # hip cannot resolve linalg ops
        original_backend = torch.backends.cuda.preferred_linalg_library()
        torch.backends.cuda.preferred_linalg_library(backend="magma")
    debug_log_tensor(f"raw_scaling_diag_matrix", raw_scaling_diag_matrix)
    
    Ux, Sx, Vx = torch.linalg.svd(raw_scaling_diag_matrix)
    Sx_pos = torch.clamp(Sx, min=1e-6)
    Y = Ux @ torch.diag(torch.sqrt(Sx_pos))
    # Y = torch.linalg.cholesky(raw_scaling_diag_matrix)
    Y = Y.to(dtype=torch.float32)
    Y_inv = torch.linalg.inv(Y)
    # Y_inv = Y_inv.to(dtype=torch.float32)
    RY = w_wq_delta @ Y
    # RY = RY.to(dtype=torch.float32)
    debug_log_tensor(f"Y", Y)
    jitter = 1e-6 * torch.randn_like(RY)
    truc_u, truc_s, truc_v = randomized_svd(A=RY+jitter,rank=rank,oversample=rank+5,n_iter=2)

    # lowrank_r = rank
    # truc_s = S[:lowrank_r]
    # truc_u = U[:, :lowrank_r]
    # truc_v = V[:lowrank_r, :]
    truc_sigma = torch.diag(truc_s)

    B = torch.matmul(truc_u, truc_sigma).to(dtype=dtype) # default to float16, check if we should save to float32
    A = torch.matmul(truc_v, Y_inv).to(dtype=dtype) # default to float16, check if we should save to float32



    # del Y, U, S, V,
    del Y, w_wq_delta, raw_scaling_diag_matrix
    del truc_s, truc_u, truc_v, truc_sigma

    # revert linalg backend
    if IS_ROCM:
        torch.backends.cuda.preferred_linalg_library(original_backend)

    return A, B
