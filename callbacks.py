import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class ClampLogStdCallback(BaseCallback):
    def __init__(self, min_std: float = 0.2, verbose: int = 0):
        super().__init__(verbose)
        self.min_log_std = np.log(min_std)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        policy = self.model.policy
        if hasattr(policy, "log_std"):
            with torch.no_grad():
                policy.log_std.clamp_(min=self.min_log_std)
