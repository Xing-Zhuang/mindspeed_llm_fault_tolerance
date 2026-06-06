from base64 import decode
from cmath import phase
from enum import Enum
from pickle import TRUE
import redis
import os
from megatron.core import parallel_state
import json
import torch
import time
from fault_tolerance.error import ErrorType
import random
from typing import Any, Dict, List, Optional, Tuple
import dataclasses
import numpy as np
import torch.distributed as dist
import random
import numbers
from collections.abc import Mapping
from .state_transfer import OptimizerTransfer,ModelWeightTransfer,OptimizerParamSchedulerTransfer
from megatron.training.initialize import initialize_megatron 
from megatron.core import parallel_state
from megatron.training.global_vars import (
    get_args,
   
)
from megatron.core import mpu
from megatron.core.utils import (
    get_model_config,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP

state_manager = None

RERUN = False

class Phase(Enum):
    INIT = "INIT"
    FB = "FB"
    UPDATE = "UPDATE"
    FINISHED = "FINISHED"

    
class StateManager:
    def __init__(
        self,
        redis_url:str,
        interval:int=1,
    ) -> None:
        # Initialize Redis client
        try:
            self.interval = interval
            self.id = os.environ['RANK'] #global rank as ID
            self.redis_client = redis.from_url(redis_url,decode_responses=True)
            self.redis_client.ping()
            print(f"RANK {os.environ['RANK']}: StateManager Connected to Redis",flush=True)
            self.set_phase(phase=Phase.INIT)
        except redis.ConnectionError:
            raise Exception("Failed to connect to Redis")

    def report_state(self):
        rank_parallel_state = {
            "global_rank": os.environ['RANK'],
            "local_rank": os.environ['LOCAL_RANK'],
            "tp_rank": parallel_state.get_tensor_model_parallel_rank(),
            "tp_size": parallel_state.get_tensor_model_parallel_world_size(),
            "pp_rank": parallel_state.get_pipeline_model_parallel_rank(),
            "pp_size": parallel_state.get_pipeline_model_parallel_world_size(),
            "dp_rank": parallel_state.get_data_parallel_rank(),
            "dp_size": parallel_state.get_data_parallel_world_size(),
        }
        key = f"{self.id}:parallel_state"
        self.redis_client.set(key, json.dumps(rank_parallel_state))


        # key = f"{self.id}:elastic_agent_id"
        # self.redis_client.set(key, os.environ['GROUP_RANK'])

    def set_phase(
        self,
        phase:Phase
    ) -> None:
        # Set the state of a key
        key = f"{self.id}:phase"
        self.redis_client.set(key, phase.value)

    def rewrap_model(self,model:Float16Module):
        """rewrap the model."""
        args = get_args()
        model = [model]
            
        DP = DDP
        config = get_model_config(model[0])

        kwargs = {}
        for f in dataclasses.fields(DistributedDataParallelConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        kwargs['grad_reduce_in_fp32'] = args.accumulate_allreduce_grads_in_fp32
        kwargs['check_for_nan_in_grad'] = args.check_for_nan_in_loss_and_grad
        kwargs['check_for_large_grads'] = args.check_for_large_grads
        if args.ddp_num_buckets is not None:
            assert args.ddp_bucket_size is None, \
                "Cannot specify both --ddp-num-buckets and --ddp-bucket-size"
            assert args.ddp_num_buckets > 0, \
                "--ddp-num-buckets must be greater than 0"
            kwargs['bucket_size'] = num_parameters // args.ddp_num_buckets
        else:
            kwargs['bucket_size'] = args.ddp_bucket_size
        kwargs['pad_buckets_for_high_nccl_busbw'] = args.ddp_pad_buckets_for_high_nccl_busbw
        kwargs['average_in_collective'] = args.ddp_average_in_collective
        if args.use_custom_fsdp and args.use_precision_aware_optimizer:
            kwargs["preserve_fp32_weights"] = False
        ddp_config = DistributedDataParallelConfig(**kwargs)

        if not getattr(args, "use_torch_fsdp2", False):
            # In the custom FSDP and DDP use path, we need to initialize the bucket size.

            # If bucket_size is not provided as an input, use sane default.
            # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
            # ring-reduce implementations are large enough to remain bandwidth-bound rather than
            # latency-bound.
            if ddp_config.bucket_size is None:
                ddp_config.bucket_size = max(
                    40000000, 1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True)
                )
            # Set bucket_size to infinity if overlap_grad_reduce is False.
            if not ddp_config.overlap_grad_reduce:
                ddp_config.bucket_size = None

        model = [DP(config=config,
                        ddp_config=ddp_config,
                        module=model_chunk,
                        # Turn off bucketing for model_chunk 2 onwards, since communication for these
                        # model chunks is overlapped with compute anyway.
                        disable_bucketing=(model_chunk_idx > 0) or args.overlap_param_gather_with_optimizer_step)
                    for (model_chunk_idx, model_chunk) in enumerate(model)]

    

        return model

    def recovery(self,model,optimizer,opt_param_scheduler,recovery_iteration):
        os.environ["FAKE_RESTART"] = "true"
        
        #0.梯度清零
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad() 

        
        #1.重建通信组
        initialize_megatron()

        #2. 重新wrap model (wrap模型时重新让wrap_model持有最新的通信组)  并更新优化器的grad_stats_parallel_group
        model = self.rewrap_model(model[0].module)
        for i, opt in enumerate(optimizer.chained_optimizers):
            setattr(opt, "grad_stats_parallel_group", parallel_state.get_model_parallel_group())
    

        #3.pre_process和post_process 同步embeddind
        gpt_model = model[0].module.module
        if gpt_model.pre_process or gpt_model.post_process:
            if parallel_state.is_rank_in_embedding_group():
                weight = gpt_model.shared_embedding_or_output_weight()
                if os.getenv("ENABLE_GLOO", "false").lower() == "true":
                    weight_data_cpu = weight.data.to('cpu')
                    torch.distributed.all_reduce(
                        weight_data_cpu, group=parallel_state.get_embedding_group()
                    )
                    weight.data = weight_data_cpu.to('cuda')
                else:
                    weight.data = weight.data.cuda()
                    torch.distributed.all_reduce(
                        weight.data, group=parallel_state.get_embedding_group()
                    )
        
        #4.发送训练状态
        dst_rank = self.get_dst_rank()
        if dst_rank is not None:
            self.send_training_state(
                model = model,
                optimizer = optimizer, 
                opt_param_scheduler = opt_param_scheduler,
                iteration=recovery_iteration,
                dst_rank=dst_rank,
            )

        
        #5.结束本次迭代，重新开始
        global RERUN
        RERUN = True       
        return None, None, None, None, None, None, None, model

    def recovery_v2(self,model,optimizer,opt_param_scheduler,recovery_iteration):
        os.environ["FAKE_RESTART"] = "true"
        
        #0.梯度清零
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad() 

        
        #1.重建通信组
        initialize_megatron()

        #2. 重新wrap model (wrap模型时重新让wrap_model持有最新的通信组)  并更新优化器的grad_stats_parallel_group
        model = self.rewrap_model(model[0].module)
        for i, opt in enumerate(optimizer.chained_optimizers):
            setattr(opt, "grad_stats_parallel_group", parallel_state.get_model_parallel_group())
    

        #3.pre_process和post_process 同步embeddind
        gpt_model = model[0].module.module
        if gpt_model.pre_process or gpt_model.post_process:
            if parallel_state.is_rank_in_embedding_group():
                weight = gpt_model.shared_embedding_or_output_weight()
                if os.getenv("ENABLE_GLOO", "false").lower() == "true":
                    weight_data_cpu = weight.data.to('cpu')
                    torch.distributed.all_reduce(
                        weight_data_cpu, group=parallel_state.get_embedding_group()
                    )
                    weight.data = weight_data_cpu.to('cuda')
                else:
                    weight.data = weight.data.cuda()
                    torch.distributed.all_reduce(
                        weight.data, group=parallel_state.get_embedding_group()
                    )
        
        #4.发送训练状态
        dst_rank = self.get_dst_rank()
        if dst_rank is not None:
            self.send_training_state(
                model = model,
                optimizer = optimizer, 
                opt_param_scheduler = opt_param_scheduler,
                iteration=recovery_iteration,
                dst_rank=dst_rank,
            )

        from megatron.core import parallel_state as ps
        while True:
            if self.redis_client.get("helper_worker_ranks") and self.redis_client.exists("helper_ranks") and self.redis_client.exists("data_receiver_ranks") and self.redis_client.exists("help_batch_size") and self.redis_client.exists("data_sender_rank"):
                          
                # tp_group = ps.get_tensor_model_parallel_group()
                # pp_group = ps.get_pipeline_model_parallel_group()
                # dp_group = ps.get_data_parallel_group()
                # tp_global_ranks = dist.get_process_group_ranks(tp_group)
                # pp_global_ranks = dist.get_process_group_ranks(pp_group)
                # dp_global_ranks = dist.get_process_group_ranks(dp_group)
                # print(f"RANK:{os.environ['RANK']} 并行组：tp_global_ranks：{tp_global_ranks}， pp_global_ranks：{pp_global_ranks},dp_global_ranks:{dp_global_ranks}")
                rank_id = int(os.environ['RANK'])

                if rank_id in json.loads(self.redis_client.get("helper_ranks")):
                    os.environ['HELPER'] = "true"
                
                if rank_id in json.loads(self.redis_client.get("helper_worker_ranks")):
                    os.environ['HELPER_WORKER'] = "true"
                
                if rank_id == int(self.redis_client.get("data_sender_rank")):
                    os.environ['DATA_SENDER'] = 'true'

                data_receiver_ranks:list[list[int]] = json.loads(self.redis_client.get("data_receiver_ranks"))
                if any(rank_id in ranks for ranks in data_receiver_ranks):
                    os.environ['DATA_RECEIVER'] = 'true'
                

                os.environ['HELP_BATCH_SIZE'] = self.redis_client.get("help_batch_size")
                os.environ['ERROR_RANK_ID'] = self.redis_client.get("error_rank_id")
                os.environ['PROXY_RANK'] = self.redis_client.get("error_rank_id")
                os.environ['DATA_SENDER_RANK'] = self.redis_client.get("data_sender_rank")
                os.environ['DATA_RECEIVER_RANKS'] = self.redis_client.get("data_receiver_ranks")
                os.environ['HELPER_WORKER_RANKS'] = self.redis_client.get("helper_worker_ranks")
                
                
                break
        
        #5.结束本次迭代，重新开始
        global RERUN
        RERUN = True       
        return None, None, None, None, None, None, None, model

    def report_error(
        self,
        error:Exception
    ) -> None:
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        self.redis_client.set(f"{self.id}:error", json.dumps(error_data))

    def wait_for_controlagent_instruction(self):
        instr_id = 0
        while True:
            key = f"instruction:train_worker:{self.id}:{instr_id}"
            if self.redis_client.exists(key):
                instr = json.loads(self.redis_client.get(key))
                if instr is not None:
                    instr_id += 1
                    yield instr
                    continue

            time.sleep(self.interval)

    def recv_training_state(
        self,
        model: Any,
        optimizer: Optional[Any],
        opt_param_scheduler: Optional[Any],
        src_rank: int,
    ) :
        #1.接收model weight
        # model_weight_transfer = ModelWeightTransfer(model=model,include_buffers=True)
        # model_weight_transfer.recv(src=src_rank)
     

        #2.接收optimizer state
        # optimizer_transfer = OptimizerTransfer(optimizer)
        # optimizer_transfer.recv(src=src_rank)

        #3.接收opt_param_scheduler
        # scheduler_transfer = OptimizerParamSchedulerTransfer(opt_param_scheduler)
        # scheduler_transfer.recv(src=src_rank)

        #4.接收iteration和num_floating_point_operations_so_far
        recv_list = [None, None]
        dist.recv_object_list(
            recv_list,
            src=src_rank,
        )

        torch.cuda.synchronize()
        print(f"rank:{os.environ['RANK']} 接收完成")

        iteration = recv_list[0]
        num_floating_point_operations_so_far = recv_list[1]
        return iteration, num_floating_point_operations_so_far
            
    def send_training_state(
        self,
        model: Any,
        optimizer: Optional[Any],
        opt_param_scheduler: Optional[Any],
        iteration: Optional[int],
        dst_rank: int,
    ) -> None:
        #1.发送model weight
        # model_weight_transfer = ModelWeightTransfer(model=model,include_buffers=True)
        # model_weight_transfer.send(dst=dst_rank)
        
        #2.发送optimizer state
        # optimizer_transfer = OptimizerTransfer(optimizer)
        # optimizer_transfer.send(dst=dst_rank)

        #3.发送opt_param_scheduler
        # scheduler_transfer = OptimizerParamSchedulerTransfer(opt_param_scheduler)
        # scheduler_transfer.send(dst=dst_rank)

        #4.发送iteration和num_floating_point_operations_so_far
        num_floating_point_operations_so_far = 0
        dist.send_object_list([iteration, num_floating_point_operations_so_far], dst=dst_rank)
        
        torch.cuda.synchronize()
        self.redis_client.set(f"train_state_send", "completed")
        print(f"rank:{os.environ['RANK']} 发送完成")

    def get_dst_rank(self):
        dst_rank = None
        while True:
            if self.redis_client.exists("redundant_rank_id") and self.redis_client.exists("error_rank_id"):
                redundant_rank_id = int(self.redis_client.get("redundant_rank_id"))
                error_rank_id = int(self.redis_client.get("error_rank_id"))
                
                
                if redundant_rank_id == int(os.environ['RANK']):
                    dst_rank = error_rank_id
                    break
                else:
                    break

        return dst_rank      

def get_state_manager() -> StateManager:
    global state_manager
    return state_manager

def init_state_manager(
    redis_url:str
) -> StateManager:
    global state_manager
    state_manager = StateManager(redis_url)
    return state_manager

