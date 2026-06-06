from enum import Enum
from functools import reduce
from tkinter import E
from typing import Any


import redis
import time
import json
import torch.distributed as dist
from datetime import timedelta

class Situation(Enum):
    FB = "FB" #都处于FB阶段
    UPDATE = "Update"  #都处于UPDATE阶段
    

class ControlAgent:
    def __init__(
        self,
        redis_url: str,
        tcp_store_host: str,
        tcp_store_port: int,
        train_worker_num: int, #world size
    ) -> None:
        try:
            self.train_worker_num = train_worker_num
            self.redis_client = redis.StrictRedis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            print("Connected to Redis at", redis_url)

            self.tcp_store_server = dist.TCPStore(
                host_name=tcp_store_host,
                port=tcp_store_port,
                world_size=None,
                is_master=True,
                timeout=timedelta(minutes=5),
                wait_for_workers=False,
                use_libuv=True,
            )
            print(f"TCPStore server started on {tcp_store_host}:{tcp_store_port}")

        except Exception as e:
            print(e)
    
    def show_all_info_in_redis(self):
        for key in self.redis_client.scan_iter("*"):
            value = self.redis_client.get(key)

            print("KEY:", key)

            try:
                print("VALUE:", json.loads(value))
            except:
                print("VALUE:", value)

            print("-" * 50)

    def monitor_until_error(self,interval:int):
        print(f"Starting monitor with interval {interval} and world size {self.train_worker_num}")
        need_monitor_ids = [i for i in range(self.train_worker_num)]
        
        #监听并收集所有rank的error
        error_info = {}
        while True:
            time.sleep(interval)
            if len(error_info) == self.train_worker_num:
                print("All ranks error have been collected")
                break
            for rank in need_monitor_ids:
                error_key = f"{rank}:error"
                if self.redis_client.exists(error_key):
                    error_data = self.redis_client.get(error_key)
                    #print(f"RANK {rank} ERROR:", error_data)
                    error_info[rank] = {"rank": rank, "error": json.loads(error_data)}

        #把其他的信息也打包放进去，便于后续处理
        for rank in need_monitor_ids:
            key = f"{rank}:phase"
            phase = self.redis_client.get(key)
            error_info[rank]["phase"] = phase

            key = f"{rank}:parallel_state"
            parallel_state = self.redis_client.get(key)
            error_info[rank]["parallel_state"] = json.loads(parallel_state)

        return error_info

    def send_instruction(self, target:str, instr: str, instr_data: dict):
        self.redis_client.set(target, json.dumps({"instr": instr, "instr_data": instr_data}))
    
    def clear_instruction(self):
        raise NotImplementedError
    
    def recovery(self, error_info: dict):
        self.redis_client.set(f"recovery_state", "start")
        situation = self.judge_situation(error_info)

        if situation == Situation.FB or situation == Situation.UPDATE:
            print(f"=============Situation: {situation}================")
            #找到error的rank
            error_rank_id = self._find_error_rank(error_info)
            self.redis_client.set(f"error_rank_id", error_rank_id)
            assert(error_rank_id is not None)

            #选择冗余rank和重启的npu
            redundant_rank_id = 2
            relaunch_node_id = 0
            relaunch_npu_id = 1
            self.redis_client.set(f"redundant_rank_id", redundant_rank_id)
            self.redis_client.set(f"relaunch_node_id", relaunch_node_id)
            self.redis_client.set(f"relaunch_npu_id", relaunch_npu_id)
            self.redis_client.set(f"relaunch_args", self.redis_client.get(f"{error_rank_id}:launch_config"))
            
            #等待发送完成
            while True:
                if self.redis_client.exists(f"train_state_send"):
                    train_state_send = self.redis_client.get(f"train_state_send")
                    if train_state_send == "completed":
                        break          

            #清理redis
            self.redis_client.delete(f"error_rank_id")
            self.redis_client.delete(f"redundant_rank_id")
            self.redis_client.delete(f"relaunch_node_id")
            self.redis_client.delete(f"relaunch_npu_id")
            self.redis_client.delete(f"relaunch_args")
            self.redis_client.delete(f"train_state_send")
            for rank in range(self.train_worker_num):
                self.redis_client.delete(f"{rank}:error")

            #设置recovery_state为completed
            self.redis_client.set(f"recovery_state", "completed")

    def recovery_v2(self, error_info: dict):
        self.redis_client.set(f"recovery_state", "start")
        situation = self.judge_situation(error_info)

        if situation == Situation.FB or situation == Situation.UPDATE:
            print(f"=============Situation: {situation}================")
            #找到error的rank
            error_rank_id = self._find_error_rank(error_info)
            self.redis_client.set(f"error_rank_id", error_rank_id)
            assert(error_rank_id is not None)


            #设置helper
            helper_ranks = [2,3,6,7]
            helper_worker_ranks = [2]
            data_receiver_ranks = [[2,3]]
            mbs = 4
            help_batch_size = mbs
            self.redis_client.set(f"helper_ranks", json.dumps(helper_ranks))
            self.redis_client.set(f"data_receiver_ranks", json.dumps(data_receiver_ranks))
            self.redis_client.set(f"help_batch_size", help_batch_size)
            self.redis_client.set(f"data_sender_rank", 0)
            self.redis_client.set(f"helper_worker_ranks", json.dumps(helper_worker_ranks))

            
            #选择冗余rank和重启的npu
            redundant_rank_id = 2
            relaunch_node_id = 0
            relaunch_npu_id = 1
            self.redis_client.set(f"redundant_rank_id", redundant_rank_id)
            self.redis_client.set(f"relaunch_node_id", relaunch_node_id)
            self.redis_client.set(f"relaunch_npu_id", relaunch_npu_id)
            self.redis_client.set(f"relaunch_args", self.redis_client.get(f"{error_rank_id}:launch_config"))

            
            
            #等待发送完成
            while True:
                if self.redis_client.exists(f"train_state_send"):
                    train_state_send = self.redis_client.get(f"train_state_send")
                    if train_state_send == "completed":
                        break          

            #清理redis
            #self.redis_client.delete(f"error_rank_id")
            self.redis_client.delete(f"redundant_rank_id")
            self.redis_client.delete(f"relaunch_node_id")
            self.redis_client.delete(f"relaunch_npu_id")
            self.redis_client.delete(f"relaunch_args")
            self.redis_client.delete(f"train_state_send")
            for rank in range(self.train_worker_num):
                self.redis_client.delete(f"{rank}:error")

            #设置recovery_state为completed
            self.redis_client.set(f"recovery_state", "completed")
        
    def judge_situation(self, error_info: dict) -> Situation:
        phases = set()

        for rank, info in error_info.items():
            if "phase" not in info:
                raise ValueError(f"rank {rank} 缺少 phase 字段")

            phases.add(info["phase"])

        if phases == {"FB"}:
            return Situation.FB

        if phases == {"UPDATE"}:
            return Situation.UPDATE

        
        raise ValueError(f"不属于任何已定义 Situation，当前 phases={phases}")

    def _find_error_rank(
        self,
        error_info: dict
    ) -> dict:
        """
        假设 error_info 中最多只有一个 NPUNotAvailable rank。

        返回：
            error_rank:
                NPUNotAvailable 的 rank。
                如果不存在，返回 None。

        """

        error_rank: None = None

        for rank, info in error_info.items():
            error_type = info.get("error", {}).get("error_type")

            if error_type == "NPUNotAvailable":
                error_rank = rank
                break

        return error_rank        
      

    def run(self):
        start_time = time.time()
        while True:
            error_info = self.monitor_until_error(interval=3)
            start_time = time.time()
            self.recovery_v2(error_info)
            print(f"本轮容错结束，耗时: {time.time() - start_time} seconds")
            
     

def main():
    control_agent = ControlAgent(
        redis_url="redis://localhost:6379",
        tcp_store_host="0.0.0.0",
        tcp_store_port=29599,
        train_worker_num=8,
    )
    control_agent.run()
    

if __name__ == "__main__":
    main()