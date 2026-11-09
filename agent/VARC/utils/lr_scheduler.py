import math
import torch
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float=0.5, last_epoch: int=-1) -> LambdaLR:

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine_lr_multiple = 0.5 * (1.0 + math.cos(math.pi * num_cycles * progress * 2))
        return max(0.0, cosine_lr_multiple)
    return LambdaLR(optimizer, lr_lambda, last_epoch)
