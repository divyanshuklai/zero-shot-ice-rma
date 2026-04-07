# Architecture & Experiment Log: Zero-Shot-Ice (RMA)

## 1. Current Architecture (Baseline Phase 0)
*   **Agent**: PPO (Stable Baselines 3 implementation).
*   **Environment**: `Ant-v5` MuJoCo wrapper configured with `unitree_go1` XML.
*   **Observation Space (30-dim)**:
    *   Gravity vector (3)
    *   Root angular velocity (3)
    *   Joint positions (12)
    *   Joint velocities (12)
*   **Action Space (12-dim)**: Target joint positions relative to a nominal "home" pose.
*   **Reward Function**: Bioenergetics-inspired (RMA Paper).
    *   `reward[0]`: Forward (Weight 20, Cap 0.35)
    *   `reward[1]`: Lateral/Rotation Penalty (Weight 21)
    *   `reward[2:10]`: Physical penalties (Work, Impact, Smoothness, etc.)
*   **Reward Scheduling**: $k$ multiplier (starts at 0.03) applied only to `reward[2:10]`.
*   **Exploration**: `min_log_std = 0.2` (clamped) with `ent_coef = 0.0`.

## 2. Observed Shortcomings (Run: `loss_rma_1`)
*   **Metric**: `mean_step_reward` stays deeply negative (~ -12 to -18).
*   **Metric**: `train.value_loss` is astronomical (10k to 110k).
*   **Behavior**: Agent fails to discover forward walking; likely collapsing or vibrating to minimize penalties.
*   **Convergence**: No signs of gait discovery after 42M steps.

## 3. Hypotheses for Failure
1.  **Normalization Hypothesis**: The range of observations and the magnitude of raw rewards/returns are too large for the PPO value network to stabilize, causing gradient explosion.
2.  **Penalty Dominance Hypothesis**: The Lateral/Rotation penalty (`reward[1]`) is unscaled by $k$. Because it is weighted heavily (21), any falling/spinning at the start creates a massive negative signal that the small `forward_reward` (capped at 7.0) cannot overcome.
3.  **Temporal Context Hypothesis**: The agent only sees the current state $x_t$. Without the previous action $a_{t-1}$, it cannot learn the oscillatory nature of a gait.
4.  **Terrain Hypothesis**: Flat terrain provides insufficient contact diversity. The agent "slides" instead of learning to use friction to push forward.

## 4. Proposed Changes (TODO)
- [ ] **Observation/Reward Normalization**: Wrap environment in `VecNormalize`.
- [ ] **Terrain Complexity**: Switch from `flat=True` to fractal terrain (`flat=False`).
- [ ] **Observation Augmentation**: Concatenate $a_{t-1}$ to the observation vector.
- [ ] **Penalty Scaling**: Test if `reward[1]` also needs to be scaled by $k$ during early training.
- [ ] **Velocity Cap**: Consider increasing forward velocity cap from 0.35 to 0.6.

## 5. Experiment Log
### Exp 1: Baseline No-Extrinsic (Failed)
*   **Status**: Terminated at 42M steps.
*   **Result**: No learning.
*   **Takeaway**: Raw rewards and lack of temporal context are likely preventing progress.
