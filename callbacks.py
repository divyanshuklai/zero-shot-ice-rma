import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

class RewardScheduleCallback(BaseCallback):
    def __init__(self, init_k = 0.03, exponent=0.997, verbose=0):
        super().__init__(verbose)
        self.k = init_k
        self.exponent = exponent

    def _on_rollout_start(self):
        #broadcast cur k
        self.training_env.env_method("set_reward_k", self.k)
        #step schedule for next iter
        self.k = self.k ** self.exponent
        if self.verbose:
            print(f"[RewardScheduleCallback] k = {self.k:.6f}")

    def  _on_step(self) -> bool:
        return True


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


class AverageStepRewardCallback(BaseCallback):
    """Logs the mean per-step reward over each rollout to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        rollout_buffer = self.model.rollout_buffer
        mean_reward = rollout_buffer.rewards.mean().item()
        self.logger.record("rollout/mean_step_reward", mean_reward)
        if self.verbose:
            print(f"[AverageStepRewardCallback] mean_step_reward = {mean_reward:.4f}")
