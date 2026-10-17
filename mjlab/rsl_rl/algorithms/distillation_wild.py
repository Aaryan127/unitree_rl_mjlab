"""Distillation with belief reconstruction (Miki et al. 2022, student training).

Rollouts are collected with the student's own actions (DAgger-style), each step is
labelled with the teacher's action, and the student is trained with

    loss = MSE(student_action, teacher_action)
         + reconstruction_coef * (MSE(decoded_height, clean_height)
                                  + MSE(decoded_privileged, privileged))

through time (truncated BPTT over `gradient_length` steps) because the student is
recurrent.
"""

from __future__ import annotations

import torch.nn as nn

from mjlab.rsl_rl.algorithms.distillation import Distillation
from mjlab.rsl_rl.modules.student_teacher_wild import StudentTeacherWild


class DistillationWild(Distillation):
  policy: StudentTeacherWild

  def __init__(self, policy, reconstruction_coef: float = 0.5, **kwargs):
    super().__init__(policy, **kwargs)
    self.reconstruction_coef = reconstruction_coef

  def update(self):
    self.num_updates += 1
    mean_behavior_loss = 0.0
    mean_reconstruction_loss = 0.0
    cnt = 0

    for _ in range(self.num_learning_epochs):
      self.policy.reset(hidden_states=self.last_hidden_states)
      self.policy.detach_hidden_states()
      loss = 0.0
      steps_in_chunk = 0
      for obs, _, teacher_actions, dones in self.storage.generator():
        actions = self.policy.act_inference(obs)
        behavior_loss = self.loss_fn(actions, teacher_actions)
        reconstruction_loss = self.policy.reconstruction_loss(obs)
        loss = loss + behavior_loss + self.reconstruction_coef * reconstruction_loss

        mean_behavior_loss += behavior_loss.item()
        mean_reconstruction_loss += reconstruction_loss.item()
        cnt += 1
        steps_in_chunk += 1

        if steps_in_chunk == self.gradient_length:
          self._gradient_step(loss)
          loss = 0.0
          steps_in_chunk = 0

        self.policy.reset(dones.view(-1))
        self.policy.detach_hidden_states(dones.view(-1))

      if steps_in_chunk > 0:  # Leftover steps of the rollout.
        self._gradient_step(loss)

    self.storage.clear()
    self.last_hidden_states = self.policy.get_hidden_states()
    self.policy.detach_hidden_states()

    return {
      "behavior": mean_behavior_loss / cnt,
      "reconstruction": mean_reconstruction_loss / cnt,
    }

  def _gradient_step(self, loss) -> None:
    self.optimizer.zero_grad()
    loss.backward()
    if self.is_multi_gpu:
      self.reduce_parameters()
    if self.max_grad_norm:
      nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
    self.optimizer.step()
    self.policy.detach_hidden_states()
