# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import json
import os
from functools import partial
from typing import Any
import torch
from megatron.training import get_args, get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core import parallel_state
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training import get_timers
import torch.distributed as dist
try:
    from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import set_post_process_flag
except ImportError:
    pass
from mindspeed_llm.training.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
from mindspeed_llm.training.utils import  set_mtp_batch_list
from mindspeed_llm.core.transformer.multi_token_prediction import generate_mtp_batch_list_on_this_tp_rank
from mindspeed.core.context_parallel.get_batch_utils import set_actual_seq_len, get_ring_degree
from mindspeed.core.context_parallel.utils import pad_data
from mindspeed_llm.tasks.posttrain.utils import compute_actual_seq_len_form_list
import torch.nn.functional as F

IGNORE_INDEX = -100


def _get_seq_len(tokens, labels, loss_mask, attention_mask, position_ids):
    if tokens is not None:
        return tokens.size(1)
    if labels is not None:
        return labels.size(1)
    if loss_mask is not None:
        return loss_mask.size(1)
    if position_ids is not None:
        return position_ids.size(1)
    if attention_mask is not None:
        return attention_mask.size(-1)
    raise RuntimeError("Cannot infer seq_len from batch tensors.")


def _get_old_batch_size(tokens, labels, loss_mask, position_ids):
    if tokens is not None:
        return tokens.size(0)
    if labels is not None:
        return labels.size(0)
    if loss_mask is not None:
        return loss_mask.size(0)
    if position_ids is not None:
        return position_ids.size(0)
    raise RuntimeError("Cannot infer old_batch_size from batch tensors.")


def _align_attention_mask(attention_mask):
    """
    把 attention_mask 对齐成 NPU FlashAttention 支持的共享 mask。

    原始:
        [old_bsz, 1, S, S]

    改成:
        [1, 1, S, S]

    这样无论 hidden_states 的 batch 是 old_bsz 还是 old_bsz + helper_bsz，
    FlashAttention 都可以接受。
    """
    if attention_mask is None:
        return None

    if attention_mask.dim() == 4:
        return attention_mask[:1, :, :, :].contiguous()

    if attention_mask.dim() == 2:
        return attention_mask.contiguous()

    raise RuntimeError(
        f"Unsupported attention_mask shape: {attention_mask.shape}"
    )


def _align_position_ids(position_ids, help_batch_size):
    """
    position_ids 可能是 None。
    如果不是 None，则复制第一条 position_ids 给 helper 样本。
    """
    if position_ids is None:
        return None

    helper_position_ids = position_ids[:1, :].expand(
        help_batch_size,
        -1,
    ).clone()

    return torch.cat([position_ids, helper_position_ids], dim=0)


def align_help_tokens_to_seq_len(
        tokens: torch.Tensor,
        help_tokens: torch.Tensor,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        tokens:        [micro_batch_size, seq_length]
        help_tokens: [helper_batch_size, helper_seq_length]

        返回:
            help_tokens: [helper_batch_size, seq_length]
        """

        target_seq_len = tokens.size(1)

        help_tokens = help_tokens.to(
            device=tokens.device,
            dtype=tokens.dtype,
        )

        # 截断到当前训练 seq_length
        help_tokens = help_tokens[:, :target_seq_len]

        # 不足则 pad 到当前训练 seq_length
        cur_seq_len = help_tokens.size(1)
        if cur_seq_len < target_seq_len:
            pad_len = target_seq_len - cur_seq_len

            pad_tokens = torch.full(
                size=(help_tokens.size(0), pad_len),
                fill_value=pad_token_id,
                dtype=help_tokens.dtype,
                device=help_tokens.device,
            )

            help_tokens = torch.cat([help_tokens, pad_tokens], dim=1)

        return help_tokens


import torch
from collections import defaultdict

def inspect_model_param_devices(
    model,
    device_filter="all",   # "all" / "cpu" / "npu"
    show_buffers=True,
    rank=None,
):
    """
    查看 model 中 parameter / buffer 所在设备。

    Args:
        model:
            torch.nn.Module
        device_filter:
            "all" -> 打印所有参数
            "cpu" -> 只打印在 CPU 上的参数
            "npu" -> 只打印在 NPU 上的参数
        show_buffers:
            是否打印 buffers
        rank:
            分布式 rank，可选
    """

    assert device_filter in ("all", "cpu", "npu"), \
        "device_filter 只能是 'all'、'cpu' 或 'npu'"

    prefix = f"[rank {rank}] " if rank is not None else ""

    def tensor_mb(t):
        return t.numel() * t.element_size() / 1024 / 1024

    def need_print(t):
        if device_filter == "all":
            return True
        return t.device.type == device_filter

    print(
        f"{prefix}"
        f"{'TYPE':<8} {'DEVICE':<10} {'DTYPE':<12} "
        f"{'SHAPE':<35} {'MB':>10} NAME"
    )
    print("-" * 120)

    total_bytes = 0
    total_tensors = 0

    for name, p in model.named_parameters():
        if not need_print(p):
            continue

        total_tensors += 1
        total_bytes += p.numel() * p.element_size()

        print(
            f"{prefix}"
            f"{'param':<8} "
            f"{str(p.device):<10} "
            f"{str(p.dtype):<12} "
            f"{str(tuple(p.shape)):<35} "
            f"{tensor_mb(p):>10.3f} "
            f"{name}"
        )

    if show_buffers:
        for name, b in model.named_buffers():
            if not need_print(b):
                continue

            total_tensors += 1
            total_bytes += b.numel() * b.element_size()

            print(
                f"{prefix}"
                f"{'buffer':<8} "
                f"{str(b.device):<10} "
                f"{str(b.dtype):<12} "
                f"{str(tuple(b.shape)):<35} "
                f"{tensor_mb(b):>10.3f} "
                f"{name}"
            )

    print("-" * 120)
    print(
        f"{prefix}"
        f"device_filter={device_filter}, "
        f"num_tensors={total_tensors}, "
        f"memory={total_bytes / 1024 / 1024:.3f} MB"
    )

class SFTTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch."""
        # Items and their type.
        keys = ['input_ids', 'attention_mask', 'labels']
        args = get_args()
        if args.reset_attention_mask:
            keys += ['position_ids', 'actual_seq_len']
        data_type = torch.int64

        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            if args.no_pad_to_seq_lengths and args.pipeline_model_parallel_size > 2:
                tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)
                return tokens, None, None, attention_mask, None
            else:
                # Broadcast data.
                data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
                # Unpack
                labels = data_b.get('labels').long()
                tokens = data_b.get('input_ids').long()
                # ignored label -100
                loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)
                if args.reset_attention_mask:
                    position_ids = data_b.get('position_ids').long()
                    batch = {
                        'tokens': tokens,
                        'labels': labels,
                        'loss_mask': loss_mask,
                        'attention_mask': None,
                        'position_ids': position_ids
                    }
                    if args.micro_batch_size > 1:
                        actual_seq_len = compute_actual_seq_len_form_list(data_b['actual_seq_len'])
                    else:
                        actual_seq_len = data_b['actual_seq_len']
                        actual_seq_len = actual_seq_len[actual_seq_len != -1].view(-1)
                    if args.attention_mask_type == 'causal' \
                            and args.context_parallel_size > 1 \
                            and args.context_parallel_algo == 'megatron_cp_algo':
                        actual_seq_len = pad_data(data_b['actual_seq_len'].view(-1), batch, args.context_parallel_size,
                                                  args.tensor_model_parallel_size)
                        actual_seq_len /= get_ring_degree()
                    set_actual_seq_len(actual_seq_len)
                    batch = {'attention_mask': None}
                else:
                    attention_mask_1d = data_b.get('attention_mask').long()
                    attention_mask = get_tune_attention_mask(attention_mask_1d)
                    batch = {'attention_mask': attention_mask}
                batch = get_batch_on_this_cp_rank(batch)
                return None, None, None, batch['attention_mask'], None

        data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
        # Unpack
        labels = data_b.get('labels').long()
        tokens = data_b.get('input_ids').long()
        attention_mask_1d = data_b.get('attention_mask').long()
        # ignored label -100
        loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)

        if get_args().spec is not None and args.spec[0] == "mindspeed_llm.tasks.models.spec.hunyuan_spec":
            input_ids = tokens
            pad_id = 127961

            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

            loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)
            attention_mask = input_ids.ne(pad_id)

            position_ids = None
            batch = {
                'tokens': input_ids,
                'labels': labels,
                'loss_mask': loss_mask,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }
        else:

            if args.reset_attention_mask:
                position_ids = data_b.get('position_ids').long()
                batch = {
                    'tokens': tokens,
                    'labels': labels,
                    'loss_mask': loss_mask,
                    'attention_mask': None,
                    'position_ids': position_ids
                }
                if args.micro_batch_size > 1:
                    actual_seq_len = compute_actual_seq_len_form_list(data_b['actual_seq_len'])
                else:
                    actual_seq_len = data_b['actual_seq_len']
                    actual_seq_len = actual_seq_len[actual_seq_len != -1].view(-1)
                if args.attention_mask_type == 'causal' \
                        and args.context_parallel_size > 1 \
                        and args.context_parallel_algo == 'megatron_cp_algo':
                    actual_seq_len = pad_data(data_b['actual_seq_len'].view(-1), batch, args.context_parallel_size,
                                              args.tensor_model_parallel_size)
                    actual_seq_len /= get_ring_degree()
                set_actual_seq_len(actual_seq_len)

                batch = get_batch_on_this_cp_rank(batch)

                return batch.values()

            attention_mask = get_tune_attention_mask(attention_mask_1d)
            position_ids = None
            batch = {
                    'tokens': tokens,
                    'labels': labels,
                    'loss_mask': loss_mask,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids
                }
                # get batch_list for mtp_block
        if args.mtp_num_layers:
            mtp_batch_list = generate_mtp_batch_list_on_this_tp_rank(batch)
            set_mtp_batch_list(mtp_batch_list)
        batch = get_batch_on_this_cp_rank(batch)
        return batch.values()

    @staticmethod
    def loss_func(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """Loss function.

        Args:
            input_tensor (torch.Tensor): Used to mask out some portions of the loss
            output_tensor (torch.Tensor): The tensor with the losses
        """
        args = get_args()
        loss_mask = input_tensor

        losses = output_tensor.float()
        loss_mask = loss_mask[..., 1:].view(-1).float()
        if args.context_parallel_size > 1:
            loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
            loss_sum = loss[0]
            loss_mask_sum = loss[1]
        else:
            loss_sum = torch.sum(losses.view(-1) * loss_mask)
            loss_mask_sum = loss_mask.sum()

        # Check individual rank losses are not NaN prior to DP all-reduce.
        if args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            if loss_sum.isnan():
                raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                 f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

        if args.calculate_per_token_loss:
            total_loss_sum = loss_sum.clone().detach()
            total_loss_mask_sum = loss_mask_sum.clone().detach()
            torch.distributed.all_reduce(total_loss_sum, group=parallel_state.get_data_parallel_group())
            torch.distributed.all_reduce(total_loss_mask_sum, group=parallel_state.get_data_parallel_group())

            return loss_sum, loss_mask_sum.to(torch.int32), {'lm loss': [total_loss_sum, total_loss_mask_sum]}
        else:
            loss = loss_sum / loss_mask_sum
            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])
            return loss, {'lm loss': averaged_loss[0]}

    def forward_step(self, data_iterator, model):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        args = get_args()
        timers = get_timers()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
            data_iterator)

        
        if os.getenv('HELPER', 'false') == 'true':
            data_receiver_ranks:list[int] = json.loads(os.environ['DATA_RECEIVER_RANKS'])

            if mpu.is_pipeline_first_stage():
                #print(f"rank:{os.environ['RANK']} 是DATA_RECEIVER")


                rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", -1))

                old_batch_size = tokens.size(0)
                seq_len = tokens.size(1)

            
                # ============================================================
                # 接收 helper_tokens
                # ============================================================

                help_batch_size = int(os.environ["HELP_BATCH_SIZE"])
                data_sender_rank = int(os.environ["DATA_SENDER_RANK"])

                help_seq_len_tensor = torch.empty(1, dtype=torch.int64)

                dist.recv(help_seq_len_tensor, src=data_sender_rank)
                #print(f"rank:{os.environ['RANK']} 接收到 help_seq_len_tensor：{help_seq_len_tensor.item()}")

                helper_seq_len = help_seq_len_tensor.item()
                os.environ["HELPER_SEQ_LEN"] = str(helper_seq_len)

                help_tokens = torch.empty(
                    size=(help_batch_size//len(data_receiver_ranks), helper_seq_len),
                    dtype=tokens.dtype,
                    device="cpu",
                )

                dist.recv(help_tokens, src=data_sender_rank)

                help_tokens = help_tokens.to(tokens.device)

                # print(
                #     f"rank:{rank} 收到 help_tokens: "
                #     f"shape={help_tokens.shape}, "
                #     f"device={help_tokens.device}, "
                #     f"dtype={help_tokens.dtype}"
                # )


                # ============================================================
                # 对齐 helpe_tokens 到当前 seq_len
                # ============================================================

                # 注意：pad_token_id 必须是合法 vocab id。
                # 不建议用 -100，因为 Megatron 的 loss 通常会先算 cross entropy 再乘 loss_mask。
                pad_token_id = 0

                help_tokens = align_help_tokens_to_seq_len(
                    tokens=tokens,
                    help_tokens=help_tokens,
                    pad_token_id=pad_token_id,
                )

                help_batch_size = help_tokens.size(0)

                assert help_tokens.size(1) == seq_len, (
                    f"help_tokens seq_len mismatch: "
                    f"help_tokens={help_tokens.shape}, seq_len={seq_len}"
                )


                # ============================================================
                # 1. 拼接 tokens
                # ============================================================

                #print(f"rank:{rank} 拼接前 tokens.shape={tokens.shape}")

                tokens = torch.cat([tokens, help_tokens], dim=0)

                #print(f"rank:{rank} 拼接后 tokens.shape={tokens.shape}")


                # ============================================================
                # 2. 拼接 labels
                # helper 样本不参与 loss，所以 labels 的具体值不重要，
                # 但必须是合法 token id，避免 cross entropy 越界。
                # ============================================================

                help_labels = torch.full(
                    size=(help_batch_size, seq_len),
                    fill_value=pad_token_id,
                    dtype=labels.dtype,
                    device=labels.device,
                )

                labels = torch.cat([labels, help_labels], dim=0)


                # ============================================================
                # 3. 拼接 loss_mask
                # helper 样本不参与 loss，所以全部置 0。
                # ============================================================

                help_loss_mask = torch.zeros(
                    size=(help_batch_size, seq_len),
                    dtype=loss_mask.dtype,
                    device=loss_mask.device,
                )

                loss_mask = torch.cat([loss_mask, help_loss_mask], dim=0)


                # ============================================================
                # 4. 拼接 position_ids
                # position_ids 可能为 None。
                # 如果为 None，说明后续模型内部会自行生成/处理 position ids，
                # 这里保持 None 即可。
                # ============================================================

                if position_ids is not None:
                    help_position_ids = position_ids[:1, :].expand(
                        help_batch_size,
                        -1,
                    ).clone()

                    position_ids = torch.cat([position_ids, help_position_ids], dim=0)
                else:
                    position_ids = None


                # ============================================================
                # 5. 对齐 attention_mask
                #
                # 原始 attention_mask 可能是 [old_batch, 1, S, S]，
                # 但拼接后 FlashAttention 看到的 B 已经变成 new_batch。
                #
                # helper 不参与 loss，且普通 causal mask 对所有样本相同，
                # 所以推荐压成 [1, 1, S, S]，这是 NPU FlashAttention 支持的。
                # ============================================================

                if attention_mask is not None:
                    if attention_mask.dim() == 4:
                        attention_mask = attention_mask[:1, :, :, :].contiguous()

                    elif attention_mask.dim() == 2:
                        # 已经是 [S, S]，NPU FlashAttention 支持，保持不变。
                        attention_mask = attention_mask.contiguous()

                    else:
                        raise RuntimeError(
                            f"Unsupported attention_mask dim: "
                            f"shape={attention_mask.shape}, dim={attention_mask.dim()}"
                        )


                # ============================================================
                # 6. 最终一致性检查
                # ============================================================

                new_batch_size = old_batch_size + help_batch_size

                assert tokens.shape == (new_batch_size, seq_len), (
                    f"tokens shape mismatch: {tokens.shape}, "
                    f"expected=({new_batch_size}, {seq_len})"
                )

                assert labels.shape == (new_batch_size, seq_len), (
                    f"labels shape mismatch: {labels.shape}, "
                    f"expected=({new_batch_size}, {seq_len})"
                )

                assert loss_mask.shape == (new_batch_size, seq_len), (
                    f"loss_mask shape mismatch: {loss_mask.shape}, "
                    f"expected=({new_batch_size}, {seq_len})"
                )

                if position_ids is not None:
                    assert position_ids.shape == (new_batch_size, seq_len), (
                        f"position_ids shape mismatch: {position_ids.shape}, "
                        f"expected=({new_batch_size}, {seq_len})"
                    )

                if attention_mask is not None:
                    if attention_mask.dim() == 4:
                        assert attention_mask.shape[0] in (1, new_batch_size), (
                            f"attention_mask batch dim mismatch: "
                            f"attention_mask={attention_mask.shape}, "
                            f"new_batch_size={new_batch_size}"
                        )

                        assert attention_mask.shape[-2:] == (seq_len, seq_len), (
                            f"attention_mask seq dim mismatch: "
                            f"attention_mask={attention_mask.shape}, "
                            f"seq_len={seq_len}"
                        )

                    elif attention_mask.dim() == 2:
                        assert attention_mask.shape == (seq_len, seq_len), (
                            f"attention_mask shape mismatch: "
                            f"attention_mask={attention_mask.shape}, "
                            f"expected=({seq_len}, {seq_len})"
                        )

                # print(
                #     f"rank:{rank} helper batch 对齐完成: "
                #     f"tokens={tokens.shape}, "
                #     f"labels={labels.shape}, "
                #     f"loss_mask={loss_mask.shape}, "
                #     f"attention_mask={attention_mask.shape if attention_mask is not None else None}, "
                # )
            elif mpu.is_pipeline_last_stage():
                help_batch_size = int(os.environ["HELP_BATCH_SIZE"])//len(data_receiver_ranks)
                pad_token_id = 0
                # ========================================================
                # last stage:
                #   1. 不拼 tokens
                #   2. 必须对齐 attention_mask
                #   3. 最好对齐 position_ids
                #   4. 必须扩 labels / loss_mask
                #      因为 last stage 会算 loss
                # ========================================================

                seq_len = _get_seq_len(
                    tokens=tokens,
                    labels=labels,
                    loss_mask=loss_mask,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

                old_batch_size = _get_old_batch_size(
                    tokens=tokens,
                    labels=labels,
                    loss_mask=loss_mask,
                    position_ids=position_ids,
                )

                new_batch_size = old_batch_size + help_batch_size

                # 1. attention_mask 对齐
                attention_mask = _align_attention_mask(attention_mask)

                # 2. position_ids 对齐
                position_ids = _align_position_ids(
                    position_ids=position_ids,
                    help_batch_size=help_batch_size,
                )

                # 3. labels 对齐
                # helper 样本不参与 loss，所以 label 内容不重要，
                # 但必须是合法 token id，不能用 -100。
                if labels is None:
                    raise RuntimeError("pipeline last stage requires labels, but labels is None.")

                helper_labels = torch.full(
                    size=(help_batch_size, seq_len),
                    fill_value=pad_token_id,
                    dtype=labels.dtype,
                    device=labels.device,
                )

                labels = torch.cat([labels, helper_labels], dim=0)

                # 4. loss_mask 对齐
                # helper 样本不参与 loss，所以全部置 0。
                if loss_mask is None:
                    raise RuntimeError("pipeline last stage requires loss_mask, but loss_mask is None.")

                help_loss_mask = torch.zeros(
                    size=(help_batch_size, seq_len),
                    dtype=loss_mask.dtype,
                    device=loss_mask.device,
                )

                loss_mask = torch.cat([loss_mask, help_loss_mask], dim=0)

                # 5. 一致性检查
                assert labels.shape == (new_batch_size, seq_len), (
                    f"labels shape mismatch: labels={labels.shape}, "
                    f"expected=({new_batch_size}, {seq_len})"
                )

                assert loss_mask.shape == (new_batch_size, seq_len), (
                    f"loss_mask shape mismatch: loss_mask={loss_mask.shape}, "
                    f"expected=({new_batch_size}, {seq_len})"
                )

                if position_ids is not None:
                    assert position_ids.shape == (new_batch_size, seq_len), (
                        f"position_ids shape mismatch: position_ids={position_ids.shape}, "
                        f"expected=({new_batch_size}, {seq_len})"
                    )

                if attention_mask is not None:
                    if attention_mask.dim() == 4:
                        assert attention_mask.shape[0] in (1, new_batch_size), (
                            f"attention_mask batch dim mismatch: "
                            f"attention_mask={attention_mask.shape}, "
                            f"new_batch_size={new_batch_size}"
                        )
                        assert attention_mask.shape[-2:] == (seq_len, seq_len), (
                            f"attention_mask seq dim mismatch: "
                            f"attention_mask={attention_mask.shape}, "
                            f"seq_len={seq_len}"
                        )
                    elif attention_mask.dim() == 2:
                        assert attention_mask.shape == (seq_len, seq_len), (
                            f"attention_mask shape mismatch: "
                            f"attention_mask={attention_mask.shape}, "
                            f"expected=({seq_len}, {seq_len})"
                        )

                # print(
                #     f"rank:{os.environ.get('RANK', -1)} last PP stage 对齐完成: "
                #     f"tokens={tokens.shape if tokens is not None else None}, "
                #     f"labels={labels.shape}, "
                #     f"loss_mask={loss_mask.shape}, "
                #     f"attention_mask={attention_mask.shape if attention_mask is not None else None}, "
                #     f"position_ids={position_ids.shape if position_ids is not None else None}"
                # )
            else:
                help_batch_size = int(os.environ["HELP_BATCH_SIZE"])//len(data_receiver_ranks)
                pad_token_id = 0
                # ========================================================
                # middle stage:
                #   1. 不拼 tokens
                #   2. 不处理 labels / loss_mask
                #   3. 必须对齐 attention_mask
                #   4. 最好对齐 position_ids
                #
                # 因为 middle stage 收到的 hidden_states batch 已经是:
                #   old_batch_size + helper_batch_size
                # 但它本地 get_batch 得到的 attention_mask 仍可能是:
                #   [old_batch_size, 1, S, S]
                # ========================================================

                seq_len = _get_seq_len(
                    tokens=tokens,
                    labels=labels,
                    loss_mask=loss_mask,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

                old_batch_size = _get_old_batch_size(
                    tokens=tokens,
                    labels=labels,
                    loss_mask=loss_mask,
                    position_ids=position_ids,
                )

                new_batch_size = old_batch_size + help_batch_size

                # 1. attention_mask 对齐
                attention_mask = _align_attention_mask(attention_mask)

                # 2. position_ids 对齐
                position_ids = _align_position_ids(
                    position_ids=position_ids,
                    help_batch_size=help_batch_size,
                )

                # 3. 一致性检查
                if position_ids is not None:
                    assert position_ids.shape == (new_batch_size, seq_len), (
                        f"position_ids shape mismatch: position_ids={position_ids.shape}, "
                        f"expected=({new_batch_size}, {seq_len})"
                    )

                if attention_mask is not None:
                    if attention_mask.dim() == 4:
                        assert attention_mask.shape[0] in (1, new_batch_size), (
                            f"attention_mask batch dim mismatch: "
                            f"attention_mask={attention_mask.shape}, "
                            f"new_batch_size={new_batch_size}"
                        )
                        assert attention_mask.shape[-2:] == (seq_len, seq_len), (
                            f"attention_mask seq dim mismatch: "
                            f"attention_mask={attention_mask.shape}, "
                            f"seq_len={seq_len}"
                        )
                    elif attention_mask.dim() == 2:
                        assert attention_mask.shape == (seq_len, seq_len), (
                            f"attention_mask shape mismatch: "
                            f"attention_mask={attention_mask.shape}, "
                            f"expected=({seq_len}, {seq_len})"
                        )

                # print(
                #     f"rank:{os.environ.get('RANK', -1)} middle PP stage 对齐完成: "
                #     f"tokens={tokens.shape if tokens is not None else None}, "
                #     f"labels={labels.shape if labels is not None else None}, "
                #     f"loss_mask={loss_mask.shape if loss_mask is not None else None}, "
                #     f"attention_mask={attention_mask.shape if attention_mask is not None else None}, "
                #     f"position_ids={position_ids.shape if position_ids is not None else None}"
                # )
                       
        elif os.getenv('DATA_SENDER', 'false') == 'true':
            #print(f"rank:{os.environ['RANK']} 是DATA_SENDER")
            
            # import time
            # inspect_model_param_devices(model, device_filter="npu")
            # time.sleep(1000)

            help_batch_size = json.loads(os.environ['HELP_BATCH_SIZE'])
            data_receiver_ranks:list[int] = json.loads(os.environ['DATA_RECEIVER_RANKS'])
            #print(f"rank:{os.environ['RANK']} is a DATAER rank, helper_batch_size:{help_batch_size}, helper_dataer_receiver_rank_ids:{data_receiver_ranks}")

            
            for ranks in data_receiver_ranks:
                for rank in ranks:
                    #print(f"rank:{os.environ['RANK']} 发送 tokens.shape：{tokens.shape}")
                    torch.distributed.send(torch.tensor(tokens.shape[1], dtype=torch.int64), dst=rank)
            
            tokens_cpu = tokens.to('cpu')
            for idx,ranks in enumerate(data_receiver_ranks):
                tokens_send = tokens_cpu[(help_batch_size//len(data_receiver_ranks)) * idx : (help_batch_size//len(data_receiver_ranks)) * (idx+1),:]
                for rank in ranks:
                    #print(f"rank:{os.environ['RANK']} 发送 tokens_send.shape：{tokens.shape}")
                    torch.distributed.send(tokens_send, dst=rank)
            
            #print(f"rank:{os.environ['RANK']} 发送tokens完成")

            
        timers('batch-generator').stop()
        #print(f"rank:{os.environ['RANK']} 拿到的tokens shape：{tokens.shape}") #[micro_batch_size, seq_length]


        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask,
                                  labels=labels)
        else:
            output_tensor = model(tokens, position_ids, attention_mask,
                                  labels=labels, loss_mask=loss_mask)

        return output_tensor, partial(self.loss_func, loss_mask)


def forward_step_in_sft_with_dualpipe(data_iterator, model, extra_block_kwargs=None):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """

    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    set_post_process_flag(model.module.module.post_process)
    tokens, labels, loss_mask, attention_mask, position_ids = SFTTrainer.get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if extra_block_kwargs is not None:
        # excute forward backward overlaping
        output_tensor, model_graph, pp_comm_output = \
            model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask,
                  extra_block_kwargs=extra_block_kwargs)
        return (output_tensor, model_graph, pp_comm_output), partial(SFTTrainer.loss_func, loss_mask)
    else:
        output_tensor, model_graph = model(
            tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
        return (output_tensor, model_graph), partial(SFTTrainer.loss_func, loss_mask)
