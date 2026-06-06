import json
import re
import redis
import time
import ast
import json
import os
import shlex
import subprocess
import redis
from typing import Any


class TrainAgent:
    def __init__(
        self,
        agent_id:int, #必须跟torchrun的elasticagent一致
        redis_url: str,
        interval: int = 1,
    ) -> None:
        try:
            self.interval = interval
            self.agent_id = agent_id
            self.redis_client = redis.from_url(redis_url,decode_responses=True)
            self.redis_client.ping()
        except Exception as e:
            print(f"Error connecting to Redis: {e}")

    def _start_process_from_config(self, config, npu_id, is_proxy=False):
        args = config["args"]

        env = config.get("env") or {}
        env = {
            str(k): str(v)
            for k, v in env.items()
        }
        env["RESTART"] = "true"
        env["RESTART_NPU_ID"] = str(npu_id)
        env["HELPER_RANKS"] = self.redis_client.get("helper_ranks")
        env["HELP_BATCH_SIZE"] = self.redis_client.get("help_batch_size")
        env["DATA_RECEIVER_RANKS"] = self.redis_client.get("data_receiver_ranks")
        env["DATA_SENDER_RANK"] = self.redis_client.get("data_sender_rank")
        env["HELPER_WORKER_RANKS"] = self.redis_client.get("helper_worker_ranks")
        if is_proxy:
            env["PROXY"] = "true"
            
        
        error_rank = int(self.redis_client.get("error_rank_id"))
        data_sender_rank = int(self.redis_client.get("data_sender_rank"))
        if error_rank == data_sender_rank:
            env["DATA_SENDER"] = "true"
        

        kwargs = dict(config.get("kwargs") or {})

        # 非常重要：
        # env["PWD"] 只是环境变量，不会真的切换工作目录。
        # 你的 args 里有 posttrain_gpt.py、./finetune_dataset/alpaca 等相对路径，
        # 所以这里必须显式设置 cwd。
        if "cwd" not in kwargs:
            pwd = env.get("PWD")
            if pwd:
                kwargs["cwd"] = pwd

        print("Launching process:")
        print("cwd:", kwargs.get("cwd"))
        print("cmd:", " ".join(shlex.quote(str(x)) for x in args))
        print("RANK:", env.get("RANK"))
        print("LOCAL_RANK:", env.get("LOCAL_RANK"))
        print("WORLD_SIZE:", env.get("WORLD_SIZE"))

        return subprocess.Popen(
            args=args,
            env=env,
            **kwargs,
        )

    def run(self):
        while True:
            if self.redis_client.exists("relaunch_node_id"):
                relaunch_node_id = self.redis_client.get("relaunch_node_id")
                if relaunch_node_id == str(self.agent_id):

                    while True:
                        if self.redis_client.exists("relaunch_args") and self.redis_client.get("relaunch_npu_id") is not None:
                            break
                    relaunch_args = json.loads(self.redis_client.get("relaunch_args"))
                    relaunch_npu_id = self.redis_client.get("relaunch_npu_id")
                    self._start_process_from_config(relaunch_args, relaunch_npu_id, is_proxy=True)

                    #等待启动完成
                    while True:
                        if self.redis_client.exists(f"recovery_state"):
                            relaunch_state = self.redis_client.get(f"recovery_state")
                            if relaunch_state == "completed":
                                break


            time.sleep(self.interval)
        



def main():
    state_agent = TrainAgent(agent_id=0, redis_url="redis://localhost:6379")
    state_agent.run()

if __name__ == "__main__":
    main()