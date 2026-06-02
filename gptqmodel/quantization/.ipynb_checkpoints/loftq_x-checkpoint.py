# loftq_x modification
import contextlib
import math
import os
import sys
import threading
import time
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.nn.modules.conv import _ConvNd
from ..looper.named_module import NamedModule
from ..quantization import QuantizeConfig
from ..utils.device import get_device
from ..utils.logger import setup_logger
from .gar import compose_final_perm, compute_global_perm, compute_local_perms, invert_perm
from .quantizer import HF_OPTIMUM, Quantizer

log = setup_logger()
lock = threading.Lock()

_WORKSPACE_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}
_WORKSPACE_LOCKS: Dict[Tuple[str, Optional[int]], threading.Lock] = {}
_BF16_SUPPORT_CACHE: Dict[Tuple[str, Optional[int]], bool] = {}


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
        
def _device_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    dev = torch.device(device)
    return dev.type, dev.index

def _workspace_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    return _device_cache_key(device)

def _needs_workspace_resize(
    workspace: Optional[torch.Tensor],
    dtype: torch.dtype,
    required_rows: int,
    cols: int,
) -> bool:
    if workspace is None:
        return True
    if workspace.ndim != 2:
        return True
    if workspace.dtype != dtype:
        return True
    if workspace.shape[1] != cols:
        return True
    if workspace.shape[0] < required_rows:
        return True
    return False

@contextlib.contextmanager
def _lease_workspace(device: torch.device, dtype: torch.dtype, cols: int, required_rows: int):
    key = _workspace_cache_key(device)
    lock = _WORKSPACE_LOCKS.setdefault(key, threading.Lock())
    with lock:
        workspace = _WORKSPACE_CACHE.pop(key, None)
        if _needs_workspace_resize(workspace, dtype, required_rows, cols):
            rows = max(required_rows, 1)
            workspace = torch.empty((rows, cols), dtype=dtype, device=device)
    try:
        yield workspace
    finally:
        with lock:
            _WORKSPACE_CACHE[key] = workspace

def _device_supports_bfloat16(device: torch.device) -> bool:
    cache_key = _device_cache_key(device)
    cached = _BF16_SUPPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    dev = torch.device(device)
    if dev.type == "meta":
        _BF16_SUPPORT_CACHE[cache_key] = False
        return False
    try:
        a = torch.zeros((1, 1), dtype=torch.bfloat16, device=dev)
        b = torch.zeros((1, 1), dtype=torch.bfloat16, device=dev)
        _ = torch.matmul(a, b)
        support = True
    except Exception:
        support = False
    _BF16_SUPPORT_CACHE[cache_key] = support
    return support

def get_number_of_rows_and_cols(layer: nn.Module):
    if isinstance(layer, NamedModule):
        layer = layer.module
    if isinstance(layer, transformers.Conv1D):
        # transformers.Conv1D: weight shape is (n_in, n_out)
        return layer.weight.shape[1], layer.weight.shape[0]
    else:
        # weight shape is (n_out, n_in)
        return layer.weight.shape[0], np.prod(layer.weight.shape[1:])

class GPTQ:  
    def __init__(self, module: nn.Module, qcfg: Optional[QuantizeConfig] = None):
        self.lock = threading.Lock()
        # 保留GPTQ原有的层结构参数
        self.rows, self.columns = get_number_of_rows_and_cols(module)
        if isinstance(module, NamedModule):
            self.module = module.module
            self.name = module.name
            self._named_module = module
        else:
            self.name = HF_OPTIMUM
            self.module = module
            self._named_module = None
        self._original_rows = self.rows
        self._original_columns = self.columns
        
        # 保留GPTQ的TP并行padding信息
        if self._named_module is not None:
            pad_info = self._named_module.state.get("tp_pad_info")
        else:
            pad_info = getattr(self.module, "_tp_pad_info", None)
        if isinstance(pad_info, dict):
            pad_cols = int(pad_info.get("pad_cols", 0) or 0)
            pad_cols = max(pad_cols, 0)
        else:
            pad_info = None
            pad_cols = 0
        self._tp_pad_info = pad_info
        self._tp_pad_cols = pad_cols
        if self._tp_pad_cols:
            self.columns += self._tp_pad_cols
        
        # 保留GPTQ的设备相关配置
        module_device = get_device(self.module)
        setattr(self.module, "target_device", module_device)
        if module_device.type == "meta":
            self._final_hessian_device_hint = torch.device("cpu")
        else:
            self._final_hessian_device_hint = torch.device(module_device)
        
        self._validate_module(self.module)
        self.qcfg = qcfg if qcfg else QuantizeConfig()
        self.module_copy = None
        self.H = None
        self.nsamples = 0
        self.quantizer = self.create_quantizer(name=self.name)
        self.fwd_counter = 0
        self.fail_safe = False
        self.H: Optional[torch.Tensor] = None
        
        self._device_hessian_partials: Dict[torch.device, torch.Tensor] = {}
        self._device_sample_counts: Dict[torch.device, int] = {}
        self._hessian_dirty: bool = False
        
        # -------------------------- loftq_x新增参数 --------------------------
        self.rank = 8
        self.max_iterations = 5  
        self.tol = 0.02 
        self.calibration_X = None  
        self.calibration_dtype = None 
        # -------------------------------------------------------------------

    @staticmethod
    def _validate_module(module):
        assert isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d,
                                   transformers.Conv1D)), f"We supports only linear and convolutional layers. actual = `{module}`"

    def create_quantizer(self, name: str) -> Quantizer:
        return Quantizer(qcfg=self.qcfg, name=name)

    def shape(self):
        if hasattr(self, "module"):
            return self.module.weight.shape
        else:
            return (0, 0)

    def _mock_hessian_inverse(self, H: torch.Tensor):
        """Mock hessian inverse for fast testing"""
        damp = self.qcfg.damp_percent
        identity = torch.eye(H.shape[0], dtype=torch.float32, device=H.device)
        return identity, damp

    def _clone_module(self, copy=True, device: torch.device = None, target_dtype: torch.dtype = None):
        if not device:
            device = self.module.weight.data.device

        clone = self.module.weight.data.to(copy=copy, device=device)
        if isinstance(self.module, _ConvNd):
            clone = clone.flatten(1)
        if isinstance(self.module, transformers.pytorch_utils.Conv1D):
            clone = clone.t()

        if self._tp_pad_cols:
            pad = torch.zeros(
                (clone.shape[0], self._tp_pad_cols),
                dtype=clone.dtype,  #  padding dtype与权重一致
                device=clone.device,
            )
            clone = torch.cat((clone, pad), dim=1)
        # 转换为目标dtype（优先使用外部指定的dtype，否则默认float32）
        if target_dtype is not None:
            clone = clone.to(dtype=target_dtype)
        else:
            clone = clone.float()
        return clone

    @staticmethod
    def _truncate_last_dim(tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.dim() == 0:
            return tensor
        trim = min(length, tensor.shape[-1])
        if trim == tensor.shape[-1]:
            return tensor
        return tensor.narrow(tensor.dim() - 1, 0, trim).contiguous()

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor, batch_index: Optional[int] = None):
        batch_token_size, xtx, device = self.process_batch(inp)
        if batch_token_size == 0 or xtx is None:
            return
        dev = torch.device(device)
        with self.lock:
            self.fwd_counter += 1
            existing = self._device_hessian_partials.get(dev)
            if existing is None:
                self._device_hessian_partials[dev] = xtx
            else:
                existing.add_(xtx)
                del xtx
            self._device_sample_counts[dev] = self._device_sample_counts.get(dev, 0) + batch_token_size
            self.nsamples += batch_token_size
            self._hessian_dirty = True

    def _preferred_staging_dtype(self, input_dtype: torch.dtype, device: torch.device) -> torch.dtype:
        device = torch.device(device)
        if not self.qcfg.hessian_use_bfloat16_staging:
            return torch.float32
        if input_dtype not in (torch.float16, torch.bfloat16):
            return torch.float32
        if not _device_supports_bfloat16(device):
            return torch.float32
        return torch.bfloat16

    def _resolve_hessian_chunk_size(self, rows: int, stage_dtype: torch.dtype) -> Optional[int]:
        if rows == 0:
            return None
        cfg_chunk = self.qcfg.hessian_chunk_size
        if cfg_chunk is not None:
            return max(1, min(cfg_chunk, rows))
        bytes_budget = self.qcfg.hessian_chunk_bytes
        if bytes_budget is not None:
            bytes_per_row = self.columns * torch.tensor([], dtype=stage_dtype).element_size()
            if bytes_per_row > 0:
                chunk_rows = bytes_budget // bytes_per_row
                if chunk_rows > 0:
                    return max(1, min(int(chunk_rows), rows))
            return 1
        return None

    @contextlib.contextmanager
    def _borrow_materialized_chunk_fp32(
        self,
        chunk: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        if rows == 0:
            yield chunk.new_zeros((0, self.columns), dtype=torch.float32)
            return
        device = chunk.device
        stage_dtype = self._preferred_staging_dtype(chunk.dtype, device)
        with _lease_workspace(device, stage_dtype, self.columns, rows) as staging_workspace:
            staging_view = staging_workspace[:rows, :]
            staging_view.copy_(chunk.to(dtype=stage_dtype))
            if stage_dtype == torch.float32:
                try:
                    yield staging_view
                finally:
                    if device.type == "cuda":
                        torch.cuda.current_stream(device).synchronize()
            else:
                with _lease_workspace(device, torch.float32, self.columns, rows) as fp32_workspace:
                    try:
                        fp32_view = fp32_workspace[:rows, :]
                        fp32_view.copy_(staging_view.to(torch.float32))
                        yield fp32_view
                    finally:
                        if device.type == "cuda":
                            torch.cuda.current_stream(device).synchronize()

    def _compute_hessian_xtx(self, matrix: torch.Tensor) -> torch.Tensor:
        rows = matrix.shape[0]
        if rows == 0:
            return torch.zeros((self.columns, self.columns), dtype=torch.float32, device=matrix.device)
        stage_dtype = self._preferred_staging_dtype(matrix.dtype, matrix.device)
        chunk_size = self._resolve_hessian_chunk_size(rows, stage_dtype)
        if chunk_size is None:
            mat32 = matrix.to(dtype=torch.float32)
            return torch.matmul(mat32.T, mat32)
        xtx_accum = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=matrix.device)
        for start in range(0, rows, chunk_size):
            rows_this = min(chunk_size, rows - start)
            source = matrix[start:start + rows_this]
            with self._borrow_materialized_chunk_fp32(source, rows_this) as materialized:
                materialized32 = materialized
                xtx_accum.add_(torch.matmul(materialized32.T, materialized32))
        return xtx_accum

    def process_batch(self, inp: torch.Tensor) -> Tuple[int, Optional[torch.Tensor], torch.device]:
        inp_device = get_device(inp)
        if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
            reshaped_inp = inp.reshape(-1, inp.shape[-1])
        else:
            if isinstance(self.module, nn.Conv1d):
                reshaped_inp = inp.reshape(
                    inp.size(0) * self.module.groups,
                    inp.size(1) // self.module.groups,
                    inp.shape[2],
                    1,
                )
                unfold = nn.Unfold(
                    self.module.kernel_size + (1,),
                    dilation=self.module.dilation + (1,),
                    padding=self.module.padding + (0,),
                    stride=self.module.stride + (1,),
                )
                reshaped_inp = unfold(reshaped_inp)
            else:
                reshaped_inp = inp.reshape(
                    inp.size(0) * self.module.groups,
                    inp.size(1) // self.module.groups,
                    inp.shape[2],
                    inp.shape[3],
                )
                unfold = nn.Unfold(
                    self.module.kernel_size,
                    dilation=self.module.dilation,
                    padding=self.module.padding,
                    stride=self.module.stride,
                )
                reshaped_inp = unfold(reshaped_inp)
            reshaped_inp = reshaped_inp.transpose(1, 2).flatten(0, 1)
        reshaped_inp = reshaped_inp.contiguous()
        if self._tp_pad_cols:
            pad = reshaped_inp.new_zeros((reshaped_inp.shape[0], self._tp_pad_cols))
            reshaped_inp = torch.cat((reshaped_inp, pad), dim=1)
                
        self.calibration_X = reshaped_inp.detach().to(device=self.module.target_device)
        self.calibration_dtype = self.calibration_X.dtype  # 记录激活的dtype（如FP16）
        # log.debug(
        #     f"loftq_x: Module {self.name} cached calibration_X. "
        #     f"shape={self.calibration_X.shape}, mean={self.calibration_X.mean().item():.6f},"
        #     f"dtype={self.calibration_dtype}"
        # )
        # debug_log_tensor(f"calibration_X for {self.name}", self.calibration_X)

        canonical_device = torch.device(inp_device)
        batch_token_size = reshaped_inp.shape[0]
        if batch_token_size == 0:
            del reshaped_inp
            return 0, None, canonical_device
        
        # 计算Hessian（OOM时fallback到CPU）
        try:
            xtx = self._compute_hessian_xtx(reshaped_inp).to(dtype=torch.float32)
        except RuntimeError as exc:
            if (
                torch.device(inp_device).type == "cuda"
                and "out of memory" in str(exc).lower()
            ):
                log.warn(
                    "loftq_x module '%s' fell back to CPU Hessian accumulation due to GPU OOM during batch processing.",
                    getattr(self, "name", "<unknown>"),
                )
                reshaped_inp_cpu = reshaped_inp.to(device=torch.device("cpu"))
                del reshaped_inp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                canonical_device = torch.device("cpu")
                xtx = self._compute_hessian_xtx(reshaped_inp_cpu).to(dtype=torch.float32)
                xtx = xtx.detach()
                del reshaped_inp_cpu
            else:
                del reshaped_inp
                raise
        else:
            xtx = xtx.detach()
            del reshaped_inp
        return batch_token_size, xtx, canonical_device

    def _select_hessian_target_device(self, requested: Optional[torch.device]) -> torch.device:
        if requested is not None:
            return torch.device(requested)
        hint = getattr(self, "_final_hessian_device_hint", None)
        if hint is not None:
            return torch.device(hint)
        if self._device_hessian_partials:
            partial_device = next(iter(self._device_hessian_partials.keys()))
            return torch.device(partial_device)
        return torch.device("cpu")

    def _materialize_global_hessian(self, target_device: Optional[torch.device] = None) -> None:
        device = self._select_hessian_target_device(target_device)
        with self.lock:
            if not self._hessian_dirty and self.H is not None:
                if self.H.device != device:
                    self.H = self.H.to(device=device)
                return
            total_samples = sum(self._device_sample_counts.values())
            reuse_buffer = (
                self.H is not None
                and self.H.shape == (self.columns, self.columns)
                and self.H.device == device
            )
            result_accum: torch.Tensor
            if reuse_buffer and self.H.dtype == torch.float32:
                result_accum = self.H
                result_accum.zero_()
            else:
                result_accum = torch.zeros(
                    (self.columns, self.columns),
                    dtype=torch.float32,
                    device=device,
                )
            if total_samples == 0:
                self.H = result_accum
                self.nsamples = 0
                self._hessian_dirty = False
                self._final_hessian_device_hint = device
                self._device_hessian_partials.clear()
                self._device_sample_counts.clear()
                return
            for partial_device, partial in self._device_hessian_partials.items():
                if partial.device != result_accum.device or partial.dtype != torch.float32:
                    tmp = partial.to(device=result_accum.device, dtype=torch.float32)
                    result_accum.add_(tmp)
                    del tmp
                else:
                    result_accum.add_(partial)
            result_accum.mul_(2.0 / float(total_samples))
            self.H = result_accum
            self.nsamples = total_samples
            self._hessian_dirty = False
            self._final_hessian_device_hint = result_accum.device
            self._device_hessian_partials.clear()
            self._device_sample_counts.clear()
            del result_accum

    def finalize_hessian(self, target_device: Optional[torch.device] = None) -> torch.Tensor:
        self._materialize_global_hessian(target_device=target_device)
        if self.H is None:
            self.H = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=self._select_hessian_target_device(target_device))
            
        # 添加damp
        damp = self.qcfg.damp_percent
        if damp <= 0 or damp >= 1:
            log.warning(f"damp {damp} 无效")
            return
        mean_diag = torch.mean(torch.diag(self.H))
        self.H.diagonal().add_(damp * mean_diag)
        # debug_log_tensor(f"Hessian for {self.name}", self.H)
        
        return self.H

    def fasterquant(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
    ):
        return self.hf_quantize(blocksize, percdamp, damp_auto_increment, group_size, actorder, static_groups)

    def hf_quantize(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
            act_group_aware: Optional[bool] = None,
    ):
        self.qcfg.group_size = group_size
        self.qcfg.damp_percent = percdamp
        self.qcfg.damp_auto_increment = damp_auto_increment
        self.qcfg.desc_act = actorder
        if act_group_aware is not None:
            self.qcfg.act_group_aware = act_group_aware
        self.qcfg._resolve_activation_ordering(actorder, act_group_aware)
        self.qcfg.static_groups = static_groups

        (Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples) = self.quantize(blocksize=blocksize)
        
        self.module.weight.data = Q.to(dtype=self.module.weight.dtype)
        return scale, zero, g_idx, duration, avg_loss, damp
    
    @torch.inference_mode()
    def hessian_inverse(self, H: torch.Tensor):
        damp = self.qcfg.damp_percent
        mean = torch.mean(torch.diag(H))
        orig_diag = H.diag().clone()
        while 0 < damp < 1:
            try:
                H.diagonal().add_(damp * mean)
                H2 = torch.linalg.cholesky(H)
                Hinv = torch.linalg.cholesky(torch.cholesky_inverse(H2), upper=True)
                H.diagonal().copy_(orig_diag)
                del H2
                break
            except torch._C._LinAlgError as e:
                H.diagonal().copy_(orig_diag)
                if self.qcfg.damp_auto_increment != 0:
                    log.warn(
                        f"loftq_x: Module `{self.name}` -> Current `damp_percent = {damp:.5f}` is too low, auto-incrementing by `{self.qcfg.damp_auto_increment:.5f}`")
                    damp += self.qcfg.damp_auto_increment
                else:
                    log.warn(
                        f"loftq_x: Module `{self.name}` -> Please increase damp or nsamples for calibration data to avoid the following quant error: current damp_percent=`{damp:.5f}`")
                    raise e
        if not (0 < damp < 1):
            log.error(
                f"loftq_x: Module `{self.name}` -> `damp_percent` must between 0 and 1. current is {damp}. Module cannot be correctly processed.")
            return None, 1.0
        return Hinv, damp
    
    @torch.inference_mode()    
    def build_hessian_from_X(self, X_new: torch.Tensor, target_device=None):
        X_backup = self.calibration_X
        dtype_backup = self.calibration_dtype

        partials_backup = self._device_hessian_partials
        counts_backup = self._device_sample_counts
        nsamples_backup = self.nsamples
        dirty_backup = self._hessian_dirty

        try:
            self._device_hessian_partials = {}
            self._device_sample_counts = {}
            self.nsamples = 0
            self._hessian_dirty = True

            self.calibration_X = X_new
            self.calibration_dtype = X_new.dtype
            self.add_batch(X_new, None)

            H = self.finalize_hessian(target_device=target_device)

        finally:
            self.calibration_X = X_backup
            self.calibration_dtype = dtype_backup

            self._device_hessian_partials = partials_backup
            self._device_sample_counts = counts_backup
            self.nsamples = nsamples_backup
            self._hessian_dirty = dirty_backup

        return H

    @torch.inference_mode()
    def randomized_svd(self, A, rank, oversample=8, n_iter=2):
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

    @torch.inference_mode()
    def quantize(self, blocksize=128):
        start = time.time()
        target_device = getattr(self.module, "target_device", None)
        self.H = self.finalize_hessian(target_device=target_device)

        if self.calibration_X is None or self.calibration_dtype is None:
            raise RuntimeError(f"loftq_x: Calibration data not found for module {self.name}")

        W_orig = self._clone_module(
            device=self.H.device,
            target_dtype=self.calibration_dtype
        ).detach() 
        if self.module_copy is None:
            W = W_orig.clone()
        else:
            W = self.module_copy.to(device=self.H.device, dtype=self.calibration_dtype).clone()
            del self.module_copy

        self.quantizer.find_params(W, weight=True)
        dead = torch.diag(self.H) == 0
        self.H[dead, dead] = 1
        W[:, dead] = 0


        X = self.calibration_X  # (N, n)
        X_t = X.t()
        X_t_fp32 = X_t.to(dtype=torch.float32)
        X_pinv = torch.linalg.pinv(X_t_fp32)
        # X_pinv = X_pinv.to(dtype=X_t.dtype)
        
        r = self.rank
        out_dim, in_dim = W.shape
        print(W.shape)
        A = torch.zeros(out_dim, r)
        B = torch.zeros(r, in_dim)
        A = A.to(W.device)
        B = B.to(W.device)
        
        min_residual_norm = float('inf')
        no_improve = 0
        patience = 3
        converged = False
        
        # --------------------------- 核心迭代循环 ---------------------------
        for t in range(self.max_iterations):
            log.info(f"----------------iteration{t+1}-------------------")
            W_equiv = W - A @ B  # (out_dim, in_dim)
            
            original_H = self.H.clone()
            (gptq_Q, scale, zero, g_idx, duration, prev_loss ,damp, self.nsamples) = self.gptq_quantize(
                blocksize=blocksize,
                W=W_equiv,
                H=original_H,
                dead=dead,
                target_dtype=self.calibration_dtype
            )
            R = (W - gptq_Q) @ X.T
            
            U, sigma, V_T = self.randomized_svd(A=R, rank=r)
            S_r = torch.diag(torch.sqrt(sigma))
            
            A = U @ S_r
            B = (S_r.float() @ V_T.float() @ X_pinv).to(W.dtype)
            # B = S_r @ V_T @ X_pinv
            
            debug_log_tensor(f"A for {self.name}&{t}", A)
            debug_log_tensor(f"B for {self.name}&{t}", B)

            E = (W - gptq_Q - A @ B).float() @ X.T.float()
            current_residual_norm = torch.norm(E, p="fro")**2
            log.info(f"ProjQ: 模块 {self.name} 迭代 {t+1} current_residual_norm: {current_residual_norm:.6f}")
            if current_residual_norm < min_residual_norm:
                min_residual_norm = current_residual_norm
                final_Q = gptq_Q 
                final_A = A.clone()  
                final_B = B.clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    converged = True
                    break
                    
        del R, U, sigma, V_T, S_r, X_pinv

        if not converged:
            log.warn(f"loftq_x: Module {self.name} did not converge after {self.max_iterations} iterations")
        else:
            log.info(f"loftq_x: Module {self.name} converged")
    
        
        duration = time.time() - start
        Q = final_Q.to(dtype=self.module.weight.dtype)
        return Q, scale, zero, g_idx, duration, current_residual_norm, damp, self.nsamples

    def gptq_quantize(self, blocksize: int, W: torch.Tensor, H: torch.Tensor, dead: torch.Tensor, target_dtype: torch.dtype) -> Tuple:
        self.quantizer.find_params(W, weight=True)
        H[dead, dead] = 1
        W[:, dead] = 0
        
        scale = []
        zero = []
        now_idx = 1
        groups = []
        perm = None
        invperm = None
        final_perm = None
   
        if self.qcfg.static_groups:
            import copy
            for i in range(0, self.columns, self.qcfg.group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i: (i + self.qcfg.group_size)], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)
        
        if self.qcfg.desc_act:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
        elif self.qcfg.act_group_aware:
            diag_h = torch.diag(H)
            local_perms, local_values = compute_local_perms(
                diag_h, self.qcfg.group_size, return_values=True
            )
            global_perm = compute_global_perm(
                diag_h,
                self.qcfg.group_size,
                precomputed_values=local_values,
            )
            del local_values
            final_perm = compose_final_perm(local_perms, global_perm, self.qcfg.group_size)
            W = W[:, final_perm]
            H = H[final_perm][:, final_perm]
        

        Losses = torch.zeros_like(W, dtype=torch.float32) 
        Q = torch.zeros_like(W, dtype=target_dtype) 
        Hinv, damp = self.hessian_inverse(H)
        
        if self.qcfg.mock_quantization or (self.fail_safe and self.fwd_counter == 0):
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1
                W1 = W[:, i1:i2]
                Q1 = torch.zeros_like(W1, dtype=target_dtype)  # Q1 dtype与激活一致
                
                # 处理分组量化参数
                if self.qcfg.group_size != -1:
                    if not self.qcfg.static_groups:
                        group_start_cols = list(range(i1, i2, self.qcfg.group_size))
                        for group_start in group_start_cols:
                            group_end = min(group_start + self.qcfg.group_size, self.columns)
                            if group_start < group_end:
                                self.quantizer.find_params(W[:, group_start:group_end], weight=True)
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                    else:
                        for i in range(count):
                            idx = i1 + i
                            if self.qcfg.desc_act:
                                idx = perm[idx]
                            self.quantizer = groups[idx // self.qcfg.group_size]
                
    
                if len(scale) > 0 and len(zero) > 0:
                    latest_scale = scale[-1]
                    latest_zero = zero[-1]
                    # 调整维度以适配广播（确保scale/zero dtype与W1一致）
                    if latest_scale.dim() == 1:
                        latest_scale = latest_scale.view(-1, 1).to(dtype=W1.dtype)
                    if latest_zero.dim() == 1:
                        latest_zero = latest_zero.view(-1, 1).to(dtype=W1.dtype)
                    maxq_val = 2 ** self.qcfg.bits - 1
                    # 对称/非对称量化（结果转为target_dtype）
                    if self.qcfg.sym:
                        Q1 = (latest_scale * torch.clamp(
                            torch.round(W1 / latest_scale),
                            -(maxq_val // 2),
                            maxq_val // 2
                        )).to(dtype=target_dtype)
                    else:
                        quantized = torch.clamp(
                            torch.round(W1 / latest_scale) + latest_zero,
                            0,
                            maxq_val
                        )
                        Q1 = (latest_scale * (quantized - latest_zero)).to(dtype=target_dtype)
                else:
                   
                    for i in range(count):
                        w = W1[:, i]
                        q = self.quantizer.quantize(w.unsqueeze(1)).flatten().to(dtype=target_dtype)
                        Q1[:, i] = q
                Q[:, i1:i2] = Q1
        else:
            
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1
                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1, dtype=target_dtype)  
                Err1 = torch.zeros_like(W1, dtype=torch.float32)  
                Losses1 = torch.zeros_like(W1, dtype=torch.float32)  
                if Hinv is not None:
                    Hinv1 = Hinv[i1:i2, i1:i2]
                
                for i in range(count):
                    w = W1[:, i]
                    if Hinv is not None:
                        d = Hinv1[i, i]
                    
                    
                    if self.qcfg.group_size != -1:
                        if not self.qcfg.static_groups:
                            if (i1 + i) % self.qcfg.group_size == 0:
                                self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + self.qcfg.group_size)], weight=True)
                            if ((i1 + i) // self.qcfg.group_size) - now_idx == -1:
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                        else:
                            idx = i1 + i
                            if self.qcfg.desc_act:
                                idx = perm[idx]
                            self.quantizer = groups[idx // self.qcfg.group_size]
                    
                    
                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten().to(dtype=target_dtype)
                    Q1[:, i] = q
                    
 
                    if Hinv is not None:
                        Losses1[:, i] = (w - q.to(dtype=torch.float32)) ** 2 / d**2
                        err1 = (w.to(dtype=torch.float32) - q.to(dtype=torch.float32)) / d
                        W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                        Err1[:, i] = err1
                
                Q[:, i1:i2] = Q1
               
                if Hinv is not None:
                    Losses[:, i1:i2] = Losses1 / 2
                    W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        
  
        if Hinv is not None:
            del Hinv
            if self.nsamples != 0:
                avg_loss = torch.sum(Losses).item() / self.nsamples
                if math.isnan(avg_loss):
                    print("Losses sum item:", torch.sum(Losses).item())
                    if self.fail_safe:
                        log.info(f"loftq_x: Failed due to `NaN` loss for `{self.name}`, use mock quantization retry")
                        self.qcfg.mock_quantization = True
                        return self.gptq_quantize(blocksize=blocksize, W=W, H=H, dead=dead, target_dtype=target_dtype)
                    else:
                        raise ValueError(f"GPTQ: Failed due to `NaN` loss for `{self.name}`, please try increasing calibration data samples or enable fail_safe=True")
            else:
                if self.fail_safe:
                    log.warn(f"GPTQ: Module `{self.name}` -> using fail safe mode")
                else:
                    log.warn(f"GPTQ: `{self.name}` is not activated due to model inference logic (MoE)")
                avg_loss = 999999999
        else:
            avg_loss = 999999999
        
    
        del Losses
        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns
        if self.qcfg.static_groups and self.qcfg.desc_act:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        
       
        if self.qcfg.desc_act:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]
        elif self.qcfg.act_group_aware:
            inv_final = invert_perm(final_perm)
            Q = Q[:, inv_final]
            inv_global_perm = invert_perm(global_perm)
            inv_global_perm_list = inv_global_perm.tolist()
            temp_scale = [scale[i] for i in inv_global_perm_list]
            scale = temp_scale
            temp_zero = [zero[i] for i in inv_global_perm_list]
            zero = temp_zero
            
        if self._tp_pad_cols:
            valid_cols = self._original_columns
            Q = Q[:, :valid_cols]
            g_idx = g_idx[:valid_cols]

        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()

        if Q.shape != self.module.weight.shape:
            Q = Q.reshape(self.module.weight.shape).to(self.module.weight.dtype)
        else:
            Q = Q.to(self.module.weight.dtype)

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            scale = self._truncate_last_dim(scale, valid_cols)
            zero = self._truncate_last_dim(zero, valid_cols)

        Q = Q.to(device=self.module.weight.data.device, non_blocking=False)

      
        duration = time.time() - time.time()
        
        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    def free(self):
        if hasattr(self, "H"):
            del self.H
        del self.quantizer
        if hasattr(self, "module_copy"):
            del self.module_copy
        if self._named_module is not None:
            self._named_module.state.pop("tp_pad_info", None)
        target = getattr(self, "module", None)
        if target is not None:
            del self.module
        if hasattr(self, "calibration_X"):
            del self.calibration_X
        if hasattr(self, "calibration_dtype"):
            del self.calibration_dtype

__all__ = ["GPTQ"]