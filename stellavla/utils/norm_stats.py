"""Action/state normalization statistics.

Serialized to `dataset_statistics.json` next to a checkpoint and restored by the
policy server, which un-normalizes the predicted action chunk with it. These must
match the statistics the checkpoint was trained with — a mismatch is silent and
looks like a badly trained policy.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class NormStats:
    """Aggregate normalization statistics for arm + gripper action and state.

    Dimensions:
      action: [7]  — EEF_pos(3) + EEF_rot(3) + gripper(1)
      state:  [9]  — EEF_pos_rel(3) + EEF_rot_rel(4) + gripper_qpos(2)

    Pooled across datasets with equal weight:
      pooled_mean = mean(means_i)
      pooled_std  = sqrt( mean(stds_i^2 + means_i^2) - pooled_mean^2 )
    """

    action_mean: np.ndarray  # [7] float32
    action_std: np.ndarray   # [7] float32
    state_mean: np.ndarray   # [9] float32
    state_std: np.ndarray    # [9] float32

    def to_dict(self) -> dict:
        return {
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(
            action_mean=np.array(d["action_mean"], dtype=np.float32),
            action_std=np.array(d["action_std"], dtype=np.float32),
            state_mean=np.array(d["state_mean"], dtype=np.float32),
            state_std=np.array(d["state_std"], dtype=np.float32),
        )
