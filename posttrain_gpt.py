import os

from mindspeed_llm import megatron_adaptor
from mindspeed_llm.tasks.posttrain.launcher import AutoTrainer
from mindspeed_llm.training.utils import auto_coverage
 

@auto_coverage
def launch():
    if os.getenv("FAULT_TOLERANCE", "false").lower() == "true":
        from fault_tolerance.state_manager import init_state_manager
        state_manager = init_state_manager(redis_url="redis://127.0.0.1:6379")
    
    trainer = AutoTrainer()
    
    if os.getenv("FAULT_TOLERANCE", "false").lower() == "true":
        from fault_tolerance.state_manager import get_state_manager
        get_state_manager().report_state()
    
    trainer.train()


if __name__ == '__main__':
    launch()