import copy
import time
from typing import Callable, Optional, Tuple
import logging
import torch
from torch.nn import Module

from ..looper.loop_processor import DTYPE_SIZE_COLUMN, MODULE_FEATURE_COLUMN, LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import CPU
from ..models.writer import (PROCESS_LOG_FWD_TIME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, PROCESS_LOG_NAME,
                             PROCESS_LOG_TIME, PROCESS_USED_MEMORY, QUANT_LOG_DAMP, QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES)
from ..quantization.config import METHOD, QuantizeConfig
from ..utils.device import get_device
from ..utils.model import create_quant_module, find_modules, pack_module
from ..utils.module_locks import parent_module_lock
from ..utils.logger import setup_logger, log_time_block
from ..quantization import ProjQ
from .gptq_processor import GPTQProcessor
log = setup_logger()

logging.info("ProjQProcessor: process_layer called")
class ProjQProcessor(GPTQProcessor):
    def __init__(self, tokenizer, qcfg: QuantizeConfig, calibration, prepare_dataset_func,
                 calibration_concat_size: Optional[int], calibration_sort: Optional[str], batch_size: int,
                 require_fwd: bool = True, calculate_w_wq_diff: bool = False):
        print("[DEBUG] ProjQProcessor instantiated!")
        self.rank = getattr(qcfg, "projq_rank", 8)
        self.projq_enable_lora = getattr(qcfg, "projq_enable_lora", False)
        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            calibration_concat_size=calibration_concat_size,
            calibration_sort=calibration_sort,
            prepare_dataset_func=prepare_dataset_func,
            batch_size=batch_size,
            require_fwd=require_fwd,
            calculate_w_wq_diff=calculate_w_wq_diff
        )
        self.inputs_cache = None
        print(f"[DEBUG] QuantizeConfig:{self.qcfg}")
    def preprocess(self, module: NamedModule, fail_safe: bool):

        if self.qcfg.dynamic_get(layer_name=module.full_name) is False:
            return

        # 克隆量化配置以支持动态覆盖
        qcfg_clone = copy.deepcopy(self.qcfg)

        # 应用动态配置覆盖
        if self.qcfg.dynamic is not None:
            qcfg_clone.bits = self.qcfg.dynamic_get(module.full_name, "bits", qcfg_clone.bits)
            qcfg_clone.sym = self.qcfg.dynamic_get(module.full_name, "sym", qcfg_clone.sym)
            qcfg_clone.mse = self.qcfg.dynamic_get(module.full_name, "mse", qcfg_clone.mse)
            qcfg_clone.group_size = self.qcfg.dynamic_get(module.full_name, "group_size", qcfg_clone.group_size)
            desc_act_override = self.qcfg.dynamic_get(module.full_name, "desc_act", None)
            if desc_act_override is not None:
                qcfg_clone.desc_act = desc_act_override
            act_group_aware_override = self.qcfg.dynamic_get(module.full_name, "act_group_aware", None)
            if act_group_aware_override is not None:
                qcfg_clone.act_group_aware = act_group_aware_override
            qcfg_clone.damp_percent = self.qcfg.dynamic_get(module.full_name, "damp_percent", qcfg_clone.damp_percent)
            qcfg_clone.static_groups = self.qcfg.dynamic_get(module.full_name, "static_groups", qcfg_clone.static_groups)
            qcfg_clone.v2 = self.qcfg.dynamic_get(module.full_name, "v2", qcfg_clone.v2)
            qcfg_clone.v2_alpha = self.qcfg.dynamic_get(module.full_name, "v2_alpha", qcfg_clone.v2_alpha)
            qcfg_clone._resolve_activation_ordering(desc_act_override, act_group_aware_override)

        # 存储当前动态配置
        self.qcfg_dynamic = qcfg_clone

        # 初始化ProjQ量化器（使用ProjQ替代GPTQ/GPTQv2）
        tmp = ProjQ(
            module=module,
            qcfg=qcfg_clone,
            rank=self.rank  # 传递ProjQ特有的秩参数
        )
        tmp.quantizer.configure(perchannel=True)
        self.tasks[module.name] = tmp
        
        print(f"[ProjQ] {module.full_name} 实际生效参数: bits={qcfg_clone.bits}, group_size={qcfg_clone.group_size}, rank={self.rank}\\\")
        
        
    def process(self, module: NamedModule):
        self.pb.title(f"Quantizing {module.name} in layer ").draw()

        with self.lock:
            projq = self.tasks[module.name]
        # -----------------------------------------------
        # 确认 projq 与 module device/dtype 匹配
        log.info(f"[ProjQProcessor] processing {module.full_name}, module.weight device={module.weight.device}, dtype={module.weight.dtype}")
        g_module = getattr(projq, "module", None)
        if g_module is not None:
            log.info(f"[ProjQProcessor] projq.module.weight device={getattr(g_module,'weight',None).device}, dtype={getattr(g_module,'weight',None).dtype}")
        # -----------------------------------------------
        expected_device = self._get_expected_device(module)
        self._validate_device_consistency(module, projq, expected_device)

        # 执行ProjQ量化过程
        wq, q_scales, q_zeros, q_g_idx, duration, avg_loss, damp_percent, nsamples = projq.quantize()
         # -----------------------------------------------
        # basic sanity
        log.info(f"[ProjQ] finished quantize for {module.full_name}: wq.shape={wq.shape}, dtype={wq.dtype}, device={wq.device}")
        assert wq.shape == module.weight.data.shape, f"shape mismatch: {wq.shape} vs {module.weight.data.shape}"
        # scales/zeros shapes and dtypes
        log.info(f"q_scales: shape={q_scales.shape}, dtype={q_scales.dtype}, min={q_scales.to(torch.float32).min().item():.6g}, max={q_scales.to(torch.float32).max().item():.6g}")
        log.info(f"q_zeros: shape={q_zeros.shape}, dtype={q_zeros.dtype}, min={q_zeros.min().item()}, max={q_zeros.max().item()}")
         # -----------------------------------------------

        # 将量化参数转移到CPU
        module.stream_state_payload_to_cpu({
            "q_scales": q_scales,
            "q_zeros": q_zeros,
            "q_g_idx": q_g_idx,
        })
        del q_scales, q_zeros, q_g_idx

     
        with self.lock:
            self.durations.append(duration)
            self.avg_losses.append(avg_loss)
            self.module_names.append(f"layer-{module.layer_index}-{module.name}")

        stat = self._build_stat_dict(module, avg_loss, nsamples, damp_percent, duration)
        with self.lock:
            self.log.append(stat)
        self.log_new_row(stat)

        if self.calculate_w_wq_diff:
            self._handle_weight_diff(module, wq)

        with self.lock:
            projq.free()
            if self.calculate_w_wq_diff:
                module.state.update({"wq": wq})

     # -----------------------------------------------
        # module.weight.data = wq
        log.info(f"[Before replace] first 8 weights orig: {module.weight.data.flatten()[:8].cpu().numpy().tolist()}")
        module.weight.data = wq
        log.info(f"[After replace] first 8 weights new: {module.weight.data.flatten()[:8].cpu().numpy().tolist()}")
        # assert device/dtype
        assert module.weight.data.device == torch.device("cpu") if self._get_expected_device(module)==torch.device("cpu") else module.weight.data.device == self._get_expected_device(module)
     # -----------------------------------------------
    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):

        module.stream_sync()
        with self.lock:
            if self.calculate_w_wq_diff:
                module.weight.data = module.state.pop("wq").to(CPU)
            module.state.pop("w", None)
            module.state.pop("w_wq_diff", None)
            q_zeros = module.state.pop("q_zeros").clone()
            q_scales = module.state.pop("q_scales").clone()
            q_g_idx = module.state.pop("q_g_idx").clone()

   
        assert q_zeros.device == CPU, "q_zeros should be on CPU"
        assert q_scales.device == CPU, "q_scales should be on CPU"
        assert q_g_idx.device == CPU, "q_g_idx should be on CPU"

   
        layers = find_modules(model.model)
        module_label = getattr(module, "full_name", module.name)
        parent_key = module.full_name


        timer = getattr(model, "quant_region_timer", None)
        create_start = time.perf_counter() if timer else None
        with log_time_block("create_quant_module", logger=log, module_name=module_label):
            with parent_module_lock(parent_key):
                create_quant_module(
                    name=module.full_name,
                    linear_cls=model.qlinear_kernel,
                    bits=self.qcfg.bits,
                    desc_act=self.qcfg.desc_act,
                    dynamic=self.qcfg.dynamic,
                    group_size=self.qcfg.group_size,
                    module=model.model,
                    submodule=module,
                    sym=self.qcfg.sym,
                    device=self.qcfg.device,
                    lm_head_name=model.lm_head,
                    pack_dtype=self.qcfg.pack_dtype,
                    register_buffers=False,
                )
        if timer and create_start:
            timer.record("submodule_finalize_create", time.perf_counter() - create_start, source=module_label)


        q_modules = {
            name: submod for name, submod in find_modules(model.model, [model.qlinear_kernel]).items()
            if name == module.full_name
        }
        pack_start = time.perf_counter() if timer else None
        with log_time_block("pack", logger=log, module_name=module_label):
            with parent_module_lock(parent_key):
                packer_label = pack_module(
                    name=module.full_name,
                    qModules=q_modules,
                    q_scales=q_scales,
                    q_zeros=q_zeros,
                    q_g_idx=q_g_idx,
                    layers=layers,
                    quant_linear_cls=model.qlinear_kernel,
                    lock=self.lock,
                    quantize_config=self.qcfg,
                )
        if timer and pack_start:
            timer.record(
                "submodule_finalize_pack",
                time.perf_counter() - pack_start,
                source=f"{module_label} [{packer_label or 'module.pack_original'}]"
            )

        
        with self.lock:
            self.result_pop(module.full_name)

        del q_scales, q_zeros, q_g_idx
        module.unregister_parameter("weight")

    def finalize(self, model: BaseQModel, **kwargs):
        if not hasattr(self, 'inputs_cache'):
            self.inputs_cache = None
        super().finalize(model=model, **kwargs)
        model.quantized = True
        print(f"[DEBUG] Before finalize: quant_method={model.quantize_config.quant_method}")
        model.quantize_config.quant_method = METHOD.PROJQ  # 可根据需要扩展ProjQ专属方法
        print(f"[DEBUG] After finalize: quant_method={model.quantize_config.quant_method}")

    def name(self) -> str:
        return f"projq (rank={self.rank})"

    
    def _get_expected_device(self, module: NamedModule) -> Optional[torch.device]:
        expected_device = getattr(module, "target_device", None)
        if expected_device is None:
            expected_device = getattr(module.module, "target_device", None)
        if expected_device is None:
            expected_device = get_device(module.module)
        return torch.device(expected_device) if expected_device else None

 
    def _validate_device_consistency(self, module: NamedModule, projq, expected_device: torch.device):
        if expected_device is None:
            return

       
        module_weight = getattr(module.module, "weight", None)
        if module_weight is not None:
            assert module_weight.device == expected_device, (
                f"Module '{module.full_name}' weight device mismatch: "
                f"{module_weight.device} vs {expected_device}"
            )


        g_module = getattr(projq, "module", None)
        g_weight = getattr(g_module, "weight", None) if g_module else None
        if g_weight is not None:
            assert g_weight.device == expected_device, (
                f"ProjQ task for '{module.full_name}' weight device mismatch: "
                f"{g_weight.device} vs {expected_device}"
            )


        g_h = getattr(projq, "H", None)
        if g_h is not None:
            assert torch.device(g_h.device) == expected_device, (
                f"ProjQ Hessian for '{module.full_name}' device mismatch: "
                f"{g_h.device} vs {expected_device}"
            )


    def _build_stat_dict(self, module: NamedModule, avg_loss, nsamples, damp_percent, duration):
        return {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_LOSS: f"{avg_loss:.10f}",
            QUANT_LOG_NSAMPLES: f"{nsamples}",
            QUANT_LOG_DAMP: f"{damp_percent:.5f}",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
            PROCESS_USED_MEMORY: self.device_memory_report(),
            "rank": self.rank,  # 增加ProjQ特有的秩参数日志
            **({"dynamic": self.qcfg.dynamic_get(layer_name=module.full_name)} 
               if self.qcfg.dynamic is not None else {})
        }


    def _handle_weight_diff(self, module: NamedModule, wq):
        w_wq_diff = module.weight.data.to(dtype=torch.float32) - wq.to(dtype=torch.float32)
        with self.lock:
            module.state.update({"w_wq_diff": w_wq_diff})