# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.networks import HiddenState
from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.utils import split_and_pad_trajectories


class L2TRolloutStorage(RolloutStorage):
    """Extended RolloutStorage for Learn-to-Teach (L2T) algorithm"""

    class Transition(RolloutStorage.Transition):
        def __init__(self) -> None:
            super().__init__()
            # For L2T: student hidden states (teacher uses hidden_states from parent)
            self.student_hidden_states: tuple[HiddenState, HiddenState] | None = None

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        
        # Additional storage for student hidden states
        self.saved_student_hidden_state_a = None
        self.saved_student_hidden_state_c = None

    def add_transitions(self, transition: Transition) -> None:
        # Call parent method to save standard transition data
        super().add_transitions(transition)
        
        # Save student hidden states if provided
        if hasattr(transition, 'student_hidden_states') and transition.student_hidden_states is not None:
            self._save_student_hidden_states(transition.student_hidden_states)

    def _save_student_hidden_states(self, hidden_states: tuple[HiddenState, HiddenState]) -> None:
        # Handle None case (first call before LSTM is initialized)
        if hidden_states == (None, None) or hidden_states[0] is None or hidden_states[1] is None:
            # If storage is not initialized yet, skip and wait for first valid hidden state
            if self.saved_student_hidden_state_a is None:
                return
            # Storage is already initialized, save zero states as placeholders
            # This corresponds to zero initialization at episode start
            else:
                for i in range(len(self.saved_student_hidden_state_a)):
                    self.saved_student_hidden_state_a[i][self.step - 1].zero_()
                    self.saved_student_hidden_state_c[i][self.step - 1].zero_()
                return
        
        hidden_state_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hidden_state_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        
        # Initialize storage if needed
        if self.saved_student_hidden_state_a is None:
            self.saved_student_hidden_state_a = [
                torch.zeros(self.observations.shape[0], *hidden_state_a[i].shape, device=self.device)
                for i in range(len(hidden_state_a))
            ]
            self.saved_student_hidden_state_c = [
                torch.zeros(self.observations.shape[0], *hidden_state_c[i].shape, device=self.device)
                for i in range(len(hidden_state_c))
            ]
        
        # Copy the states (use self.step - 1 because parent already incremented step)
        for i in range(len(hidden_state_a)):
            self.saved_student_hidden_state_a[i][self.step - 1].copy_(hidden_state_a[i])
            self.saved_student_hidden_state_c[i][self.step - 1].copy_(hidden_state_c[i])

    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # Reshape teacher hidden states
                # Original shape: [time, num_layers, num_envs, hidden_dim])
                last_was_done = last_was_done.permute(1, 0)
                # Take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                hidden_state_a_batch = [
                    saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_state in self.saved_hidden_state_a
                ]
                hidden_state_c_batch = [
                    saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_state in self.saved_hidden_state_c
                ]
                # Remove the tuple for GRU
                hidden_state_a_batch = (
                    hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else hidden_state_a_batch
                )
                hidden_state_c_batch = (
                    hidden_state_c_batch[0] if len(hidden_state_c_batch) == 1 else hidden_state_c_batch
                )

                # Extract student hidden states if available
                student_hidden_state_a_batch = None
                student_hidden_state_c_batch = None
                if self.saved_student_hidden_state_a is not None:
                    student_hidden_state_a_batch = [
                        saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_student_hidden_state_a
                    ]
                    student_hidden_state_c_batch = [
                        saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for saved_hidden_state in self.saved_student_hidden_state_c
                    ]
                    student_hidden_state_a_batch = (
                        student_hidden_state_a_batch[0] if len(student_hidden_state_a_batch) == 1 else student_hidden_state_a_batch
                    )
                    student_hidden_state_c_batch = (
                        student_hidden_state_c_batch[0] if len(student_hidden_state_c_batch) == 1 else student_hidden_state_c_batch
                    )

                # Yield the mini-batch (extended format with student hidden states)
                yield (
                    obs_batch,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        hidden_state_a_batch,
                        hidden_state_c_batch,
                    ),
                    masks_batch,
                    # For L2T: student hidden states (None if not available)
                    (
                        student_hidden_state_a_batch,
                        student_hidden_state_c_batch,
                    ),
                )

                first_traj = last_traj
