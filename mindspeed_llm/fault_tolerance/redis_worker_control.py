
import json
import os
import threading
import time
from dataclasses import dataclass, field

import redis

@dataclass
class WorkerControlState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending_cmds: list[dict] = field(default_factory=list)
    last_cmd_id: int = 0
    stop: bool = False

    def push_cmd(self, cmd: dict):
        with self.lock:
            cmd_id = int(cmd["cmd_id"])
            if cmd_id <= self.last_cmd_id:
                return
            self.pending_cmds.append(cmd)
            self.last_cmd_id = cmd_id

    def pop_all(self) -> list[dict]:
        with self.lock:
            out = self.pending_cmds
            self.pending_cmds = []
            return out
            
class RedisWorkerControl:
    def __init__(self):
        self.redis_url = os.environ["REDIS_URL"]
        self.run_id = os.environ["RUN_ID"]
        self.rank = int(os.environ["RANK"])

        self.r = redis.Redis.from_url(self.redis_url, decode_responses=True)
        self.stream = f"{self.run_id}:rank:{self.rank}:cmd"
        self.state = WorkerControlState()

        self.last_stream_id = "0-0"

    def start_listener(self):
        t = threading.Thread(target=self._listen_loop, daemon=True)
        t.start()
        return t

    def _listen_loop(self):
        while not self.state.stop:
            resp = self.r.xread(
                {self.stream: self.last_stream_id},
                block=1000,
                count=16,
            )

            if not resp:
                continue

            for _, messages in resp:
                for msg_id, fields in messages:
                    self.last_stream_id = msg_id
                    payload = json.loads(fields["payload"])
                    self.state.push_cmd(payload)

    def ack(self, *, epoch: int, phase: str, payload: dict | None = None):
        key = f"xz:{self.run_id}:ack:{epoch}:{phase}"
        value = dict(payload or {})
        value["rank"] = self.rank
        value["time"] = time.time()

        self.r.hset(key, str(self.rank), json.dumps(value))
        self.r.expire(key, 24 * 3600)