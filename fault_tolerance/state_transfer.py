# optimizer_transfer.py

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
import os
try:
    import torch_npu  # noqa: F401
except ImportError:
    # 部分 torch_npu 版本不要求显式 import。
    pass


class ModelWeightTransfer:
    def __init__(self,model,include_buffers):
        self.model = model
        self.include_buffers = include_buffers
    
    def _unwrap_model(self) -> Any:
        return self.model[0].module.module

    def _iter_model_tensors(self):
        m = self._unwrap_model()

        for name, p in m.named_parameters():
            yield name, p

        if self.include_buffers:
            for name, b in m.named_buffers():
                yield name, b

    def send(self,dst):
        for name, tensor in self._iter_model_tensors():
            t = tensor.detach()

            # 非 contiguous tensor 建议先 contiguous
            send_buf = t.contiguous()
            dist.send(send_buf, dst=dst)


    def recv(self,src):
        for name, tensor in self._iter_model_tensors():
            # 如果目标 tensor 本身 contiguous，可直接 recv 到 tensor.data
            if tensor.is_contiguous():
                dist.recv(tensor, src=src)
            else:
                tmp = torch.empty_like(tensor, memory_format=torch.contiguous_format)
                dist.recv(tmp, src=src)
                tensor.copy_(tmp)

class OptimizerTransfer:
    """
    NPU/HCCL 版本的 Megatron ChainedOptimizer 点对点传输器。

    设计目标：
      - 发送端先发送 optimizer state_dict 的“结构骨架 metadata”；
      - 接收端根据发送端 metadata 构造完整骨架；
      - 再逐个接收真实 tensor 数值；
      - 最后调用接收端本地 optimizer.load_state_dict(...)。

    重要：
      接收端 optimizer 可能是刚重新拉起的训练进程，Adam state 还没有初始化。
      因此接收端不能依赖自己的 optimizer.state_dict() 来构造 state 结构。
      这里完全以发送端 metadata 为准。
    """

    _FORMAT = "megatron_chained_adam_optimizer_npu_transfer_v2"
    _TENSOR_PLACEHOLDER = "__optimizer_tensor_placeholder_v2__"

    def __init__(self, optimizer: Any):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized.")

        self.optimizer = optimizer
        self.device = self._get_current_npu_device()

        self._assert_hccl_backend()
        self._assert_chained_adam_optimizer()

    def send(self, dst: int) -> Dict[str, Any]:
        """
        发送端入口。

        流程：
          1. 从发送端 optimizer 提取完整 state_dict；
          2. 把 state_dict 里的 tensor 替换成 metadata 占位符；
          3. 先发送结构 metadata；
          4. 再按 metadata 中的顺序发送 tensor 数值。
        """
        sender_state_dict = self.optimizer.state_dict()

        # 关键点：
        # 这里生成的是“发送端真实 optimizer state 的骨架”。
        # Adam 走过 step 后才会出现 exp_avg、exp_avg_sq、step 等 state，
        # 所以这个结构必须以发送端为准。
        structure, tensors = self._extract_structure_and_tensors(sender_state_dict)

        header = {
            "format": self._FORMAT,
            "top_optimizer_class": (
                f"{type(self.optimizer).__module__}."
                f"{type(self.optimizer).__qualname__}"
            ),
            "num_chained_optimizers": len(self.optimizer.chained_optimizers),
            "structure": structure,
            "num_tensors": len(tensors),
        }

        self._send_python_obj(header, dst)

        # tensor 的发送顺序必须和 metadata 占位符中的 idx 完全一致。
        for tensor in tensors:
            self._send_tensor(tensor, dst)

        return {
            "dst": dst,
            "num_chained_optimizers": header["num_chained_optimizers"],
            "num_tensors": len(tensors),
        }

    def recv(self, src: int) -> Dict[str, Any]:
        """
        接收端入口。

        流程：
          1. 先接收发送端 metadata；
          2. 根据发送端 metadata 构造完整 optimizer state_dict 骨架；
          3. 逐个接收 tensor，填入骨架里的 tensor buffer；
          4. 调用本地 optimizer.load_state_dict(...)。

        注意：
          这里不会读取接收端 optimizer.state_dict() 的 state 结构。
          接收端即使是刚启动、Adam state 为空，也可以被发送端结构填补。
        """
        header = self._recv_python_obj(src)

        if not isinstance(header, dict):
            raise RuntimeError(f"Invalid optimizer metadata type: {type(header)}")

        if header.get("format") != self._FORMAT:
            raise RuntimeError(
                f"Unexpected optimizer metadata format: {header.get('format')}"
            )

        local_num = len(self.optimizer.chained_optimizers)
        remote_num = header["num_chained_optimizers"]

        # ChainedOptimizer 的链路数量仍然必须一致。
        # 允许不同的是 optimizer state 是否已经 lazy 初始化，
        # 不是允许模型参数组或 optimizer 链路拓扑不同。
        if local_num != remote_num:
            raise RuntimeError(
                f"Chained optimizer count mismatch: local={local_num}, remote={remote_num}"
            )

        # 关键点：
        # 这里完全根据发送端 metadata materialize 一个新的 state_dict。
        # 接收端本地 optimizer 只在最后 load_state_dict 时使用。
        receiver_state_dict, recv_buffers = self._build_state_skeleton_from_metadata(
            header["structure"]
        )

        if len(recv_buffers) != header["num_tensors"]:
            raise RuntimeError(
                f"Tensor count mismatch: materialized={len(recv_buffers)}, "
                f"remote={header['num_tensors']}"
            )

        # recv_buffers 已经被嵌入到 receiver_state_dict 对应位置。
        # 这里 recv 到 buffer 后，state_dict 里的 tensor 数值也同步被填好了。
        for buf in recv_buffers:
            self._recv_tensor(buf, src)

        # 最后让 PyTorch/Megatron optimizer 自己完成 param id 到本地 param object 的映射。
        self.optimizer.load_state_dict(receiver_state_dict)

        return {
            "src": src,
            "num_chained_optimizers": remote_num,
            "num_tensors": len(recv_buffers),
        }

    def _assert_hccl_backend(self) -> None:
        backend = str(dist.get_backend()).lower()
        if "hccl" not in backend:
            raise RuntimeError(
                f"OptimizerTransfer is NPU/HCCL-only, but current backend is {backend!r}."
            )

    def _get_current_npu_device(self) -> torch.device:
        if not hasattr(torch, "npu"):
            raise RuntimeError(
                "torch.npu is unavailable. Please check torch_npu installation."
            )

        if hasattr(torch.npu, "is_available") and not torch.npu.is_available():
            raise RuntimeError("NPU is not available.")

        idx = torch.npu.current_device()
        if isinstance(idx, torch.device):
            return idx

        return torch.device(f"npu:{idx}")

    def _assert_chained_adam_optimizer(self) -> None:
        if not hasattr(self.optimizer, "chained_optimizers"):
            raise TypeError(
                "Expected Megatron ChainedOptimizer-like optimizer with "
                "`chained_optimizers` attribute."
            )

        for idx, megatron_opt in enumerate(self.optimizer.chained_optimizers):
            base_opt = self._unwrap_base_optimizer(megatron_opt)

            if base_opt is None:
                continue

            base_name = type(base_opt).__name__.lower()
            if "adam" not in base_name:
                raise TypeError(
                    f"Only Adam-family inner optimizers are supported now. "
                    f"Got chained optimizer #{idx}: {type(base_opt)}"
                )

    @staticmethod
    def _unwrap_base_optimizer(opt: Any) -> Any:
        """
        Megatron optimizer 外面可能包了多层 wrapper。
        这里顺着 `.optimizer` 向里拆，直到拿到底层 Adam/FusedAdam。
        """
        cur = opt
        visited = set()

        while hasattr(cur, "optimizer"):
            if id(cur) in visited:
                break

            visited.add(id(cur))

            nxt = getattr(cur, "optimizer")
            if nxt is None or nxt is cur:
                break

            cur = nxt

        return cur

    def _extract_structure_and_tensors(
        self,
        obj: Any,
    ) -> Tuple[Any, List[torch.Tensor]]:
        """
        把发送端 state_dict 拆成：
          - structure: 不含真实 tensor 数值的结构骨架；
          - tensors: 真实 tensor 列表。

        structure 中每个 tensor 会被替换为一个 placeholder，例如：
          {
              "__optimizer_tensor_placeholder_v2__": True,
              "idx": 0,
              "shape": [1024, 4096],
              "dtype": "float32"
          }

        接收端会根据这些 placeholder 创建空 tensor buffer。
        """
        tensors: List[torch.Tensor] = []

        def visit(x: Any) -> Any:
            if torch.is_tensor(x):
                if x.is_sparse:
                    raise TypeError("Sparse optimizer-state tensors are not supported.")

                idx = len(tensors)
                tensors.append(x)

                return {
                    self._TENSOR_PLACEHOLDER: True,
                    "idx": idx,
                    "shape": list(x.shape),
                    "dtype": self._dtype_to_str(x.dtype),
                }

            if isinstance(x, dict):
                return {k: visit(v) for k, v in x.items()}

            if isinstance(x, list):
                return [visit(v) for v in x]

            if isinstance(x, tuple):
                return tuple(visit(v) for v in x)

            # 非 tensor 对象，例如 param_groups 里的 lr、betas、weight_decay 等，
            # 直接进入 metadata，随第一阶段一起发送。
            return x

        return visit(obj), tensors

    def _build_state_skeleton_from_metadata(
        self,
        structure: Any,
    ) -> Tuple[Any, List[torch.Tensor]]:
        """
        根据发送端 metadata 构造接收端 state_dict 骨架。

        这是接收端最关键的步骤：
          - 不看接收端当前 optimizer.state_dict()；
          - 完全按发送端结构创建 dict/list/tuple；
          - 遇到 tensor placeholder 时，在当前 NPU 上创建空 tensor；
          - 创建出来的 tensor 同时放进 recv_buffers，用于后续 dist.recv。
        """
        recv_buffers: List[torch.Tensor] = []

        def visit(x: Any) -> Any:
            if isinstance(x, dict) and x.get(self._TENSOR_PLACEHOLDER, False):
                idx = x["idx"]
                expected_idx = len(recv_buffers)

                if idx != expected_idx:
                    raise RuntimeError(
                        f"Tensor placeholder order mismatch: got idx={idx}, "
                        f"expected idx={expected_idx}"
                    )

                buf = torch.empty(
                    tuple(x["shape"]),
                    dtype=self._str_to_dtype(x["dtype"]),
                    device=self.device,
                )

                recv_buffers.append(buf)

                # 这个 buf 会被嵌入到 state_dict 对应位置。
                # 后续 dist.recv(buf, ...) 会原地填充它的数值。
                return buf

            if isinstance(x, dict):
                return {k: visit(v) for k, v in x.items()}

            if isinstance(x, list):
                return [visit(v) for v in x]

            if isinstance(x, tuple):
                return tuple(visit(v) for v in x)

            return x

        return visit(structure), recv_buffers

    def _send_python_obj(self, obj: Any, dst: int) -> None:
        """
        发送小体积 Python metadata。

        不使用 dist.send_object_list，是为了避免 object 通信和 tensor 通信混杂。
        这里把 metadata pickle 成 bytes，再包装成 NPU tensor 发送。
        """
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        num_bytes = len(payload)

        length = torch.tensor([num_bytes], dtype=torch.long, device=self.device)
        dist.send(length, dst=dst)

        if num_bytes == 0:
            return

        cpu_payload = self._bytes_to_cpu_uint8_tensor(payload)
        npu_payload = cpu_payload.to(self.device)
        dist.send(npu_payload, dst=dst)

    def _recv_python_obj(self, src: int) -> Any:
        """
        接收 Python metadata。

        先收长度，再按长度收 bytes tensor，最后 pickle.loads 还原结构。
        """
        length = torch.empty((1,), dtype=torch.long, device=self.device)
        dist.recv(length, src=src)

        num_bytes = int(length.cpu().item())

        if num_bytes < 0:
            raise RuntimeError(f"Invalid metadata byte length: {num_bytes}")

        if num_bytes == 0:
            return None

        npu_payload = torch.empty((num_bytes,), dtype=torch.uint8, device=self.device)
        dist.recv(npu_payload, src=src)

        payload = npu_payload.cpu().numpy().tobytes()
        return pickle.loads(payload)

    def _send_tensor(self, tensor: torch.Tensor, dst: int) -> None:
        """
        发送一个 optimizer state tensor。

        metadata 中已经包含 shape/dtype，所以这里不再额外发送 shape 信息。
        """
        if tensor.numel() == 0:
            return

        send_buf = tensor.detach()

        if not send_buf.is_contiguous():
            send_buf = send_buf.contiguous()

        if send_buf.device != self.device:
            send_buf = send_buf.to(self.device)

        dist.send(send_buf, dst=dst)

    @staticmethod
    def _recv_tensor(buf: torch.Tensor, src: int) -> None:
        """
        接收一个 optimizer state tensor。

        buf 是根据发送端 metadata 创建的空 tensor。
        dist.recv 会原地写入真实数值。
        """
        if buf.numel() == 0:
            return

        dist.recv(buf, src=src)

    @staticmethod
    def _bytes_to_cpu_uint8_tensor(payload: bytes) -> torch.Tensor:
        """
        把 bytes 包成 uint8 tensor。
        bytearray 是为了避免 frombuffer 对不可写 buffer 的 warning。
        """
        if hasattr(torch, "frombuffer"):
            return torch.frombuffer(bytearray(payload), dtype=torch.uint8)

        return torch.tensor(list(payload), dtype=torch.uint8)

    @staticmethod
    def _dtype_to_str(dtype: torch.dtype) -> str:
        return str(dtype).replace("torch.", "")

    @staticmethod
    def _str_to_dtype(name: str) -> torch.dtype:
        if name.startswith("torch."):
            name = name.replace("torch.", "", 1)

        if not hasattr(torch, name):
            raise ValueError(f"Unsupported tensor dtype in optimizer state: {name}")

        return getattr(torch, name)
 
class OptimizerParamSchedulerTransfer:
    """
    Megatron OptimizerParamScheduler 的 NPU/HCCL 点对点传输类。

    适用对象：
        megatron.core.optimizer_param_scheduler.OptimizerParamScheduler

    这里不做复杂 metadata 处理，因为 scheduler.state_dict() 本身很小，
    通常只有 lr/wd 配置和 num_steps。
    """

    def __init__(self, scheduler: Any):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized.")

        self.scheduler = scheduler
        self.device = self._get_current_npu_device()

    def send(self, dst: int) -> Dict[str, Any]:
        """
        发送端调用。

        直接发送 scheduler.state_dict()。
        典型内容：
            max_lr
            lr_warmup_steps
            num_steps
            lr_decay_style
            lr_decay_steps
            min_lr
            start_wd
            end_wd
            wd_incr_style
            wd_incr_steps
        """
        state_dict = self.scheduler.state_dict()

        self._send_obj(state_dict, dst)
     

    def recv(self, src: int) -> Dict[str, Any]:
        """
        接收端调用。

        接收发送端的 scheduler.state_dict()，然后 load 到本地 scheduler。

        关键点：
            Megatron 的 OptimizerParamScheduler.load_state_dict(...)
            内部通常会调用 step(increment=num_steps)。

            所以接收端 load 前先把 self.scheduler.num_steps 清零，
            避免本地已有步数和发送端 num_steps 发生叠加。
        """
        state_dict = self._recv_obj(src)

        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Invalid scheduler state_dict type: {type(state_dict)}")

        if "num_steps" not in state_dict:
            raise RuntimeError(f"Invalid scheduler state_dict: missing num_steps, {state_dict}")

        # 关键注释：
        # 接收端可能是重新拉起的进程，num_steps 通常是 0；
        # 但为了防止重复加载或异常场景，这里显式清零。
        if hasattr(self.scheduler, "num_steps"):
            self.scheduler.num_steps = 0

        # 关键注释：
        # load_state_dict 会恢复 scheduler 状态，并刷新 optimizer.param_groups
        # 里的 lr / weight_decay。
        self.scheduler.load_state_dict(state_dict)

    def _send_obj(self, obj: Any, dst: int) -> None:
        """
        把 Python 对象 pickle 成 bytes，然后通过 NPU tensor 发送。

        发送顺序：
            1. 发送 payload 长度；
            2. 发送 payload 内容。
        """
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        num_bytes = len(payload)

        length_tensor = torch.tensor([num_bytes], dtype=torch.long, device=self.device)
        dist.send(length_tensor, dst=dst)

        if num_bytes == 0:
            return

        payload_tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        payload_tensor = payload_tensor.to(self.device)

        dist.send(payload_tensor, dst=dst)

    def _recv_obj(self, src: int) -> Any:
        """
        接收 _send_obj 发来的 Python 对象。
        """
        length_tensor = torch.empty((1,), dtype=torch.long, device=self.device)
        dist.recv(length_tensor, src=src)

        num_bytes = int(length_tensor.cpu().item())

        if num_bytes < 0:
            raise RuntimeError(f"Invalid payload length: {num_bytes}")

        if num_bytes == 0:
            return None

        payload_tensor = torch.empty((num_bytes,), dtype=torch.uint8, device=self.device)
        dist.recv(payload_tensor, src=src)

        payload = payload_tensor.cpu().numpy().tobytes()

        return pickle.loads(payload)

    @staticmethod
    def _get_current_npu_device() -> torch.device:
        if not hasattr(torch, "npu"):
            raise RuntimeError("torch.npu is unavailable. Please check torch_npu.")

        if hasattr(torch.npu, "is_available") and not torch.npu.is_available():
            raise RuntimeError("NPU is not available.")

        idx = torch.npu.current_device()

        if isinstance(idx, torch.device):
            return idx

        return torch.device(f"npu:{idx}")