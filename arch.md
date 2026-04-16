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

## 6. Teacher Architecture (Phase 1 - Proposed)
*   **Privileged Information Vector ($e_t \in \mathbb{R}^{17}$)**:
    1.  Friction (1)
    2.  Mass (1)
    3.  Center of Mass (2: x, y)
    4.  Motor Strength (12: one per joint)
    5.  Discretized Terrain Height (1)
*   **Extrinsic Encoder ($\mu$)**:
    *   Input: $e_t$ (17)
    *   Architecture: MLP (256, 128) $\to$ Latent $z_t$ (8)
*   **Base Policy ($\pi$)**:
    *   Input: $[x_t, a_{t-1}, z_t]$ (Total 50)
    *   Architecture: MLP (128, 128) $\to$ Actions (12)
*   **Training Strategy**: Joint end-to-end training. The encoder $\mu$ learns to extract features that optimize the RL return.

## 7. Next Steps
1.  **Environment Privileged Wrapper**: Modify `RMAEnv` to calculate and return $e_t$ in a `Dict` observation space.
2.  **Motor Strength Implementation**: Add torque scaling to the domain randomization logic.
3.  **Local Height Calculation**: Implement foot-specific terrain height sensing in MuJoCo.

"Environmental Variations: All environmental variations with
their ranges are listed in Table I. Of these, et includes mass and
its position on the robot (3 dims), motor strength (12 dims),
friction (scalar) and local terrain height (scalar), making it a
17-dim vector. Note that although the difficulty of the terrain
profile is fixed, the local terrain height changes as the agent moves. We discretize the terrain height under each foot to the
first decimal place and then take the maximum among the four
feet to get a scalar"