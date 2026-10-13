"""Student trainer (imitation + reconstruction) scaffold.

This script collects rollouts using a frozen teacher policy and trains a student
belief policy using behavior cloning + reconstruction loss as described in the
refactoring instructions.

NOTES:
- This is a scaffold: fill model/args wiring to match your env and teacher
  checkpoint formats. See README section in `.agent/reframed_instructions.md`.
# REFACTOR: 10.2
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, list_tasks, load_runner_cls
from dataclasses import asdict
from mjlab.rl import RslRlVecEnvWrapper

# runner
from mjlab.rsl_rl.runners.on_policy_runner_wild import OnPolicyRunnerWild

# Models
from mjlab.rsl_rl.networks.belief_encoder import RecurrentAttentionPolicy
from mjlab.rsl_rl.networks.teacher_mlp import Teacher_wild


@dataclass
class StudentTrainerCfg:
    task_id: str
    teacher_ckpt: str | None = None
    seq_len: int = 50
    batch_size: int = 64
    lr: float = 1e-4
    device: str = "cpu"
    num_steps_collect: int = 10000
    max_epochs: int = 200


class ReplayBuffer:
    """Minimal sequence buffer for fixed-length sequences.

    Stores tuples of (o_p, o_e_noisy, teacher_action, s_p) per timestep and
    provides minibatches of sequences of length `seq_len`.
    """

    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        # Each entry is a list of per-step tensors shaped [num_envs, dim]
        self.buff: Dict[str, List[torch.Tensor]] = {
            "o_p": [],
            "o_e": [],
            "action": [],
            "s_p": [],
        }

    def add_step(self, o_p, o_e, action, s_p):
        # Expect inputs shaped [num_envs, dim]
        self.buff["o_p"].append(o_p.detach().cpu())
        self.buff["o_e"].append(o_e.detach().cpu())
        self.buff["action"].append(action.detach().cpu())
        self.buff["s_p"].append(s_p.detach().cpu())

    def size(self) -> int:
        return len(self.buff["o_p"])

    def sample_sequences(self, batch_size: int) -> Dict[str, torch.Tensor]:
        # Sample sequences across time and envs.
        T = self.seq_len
        S = self.size()
        if S < T:
            raise RuntimeError("Not enough steps in buffer to sample a sequence")
        num_envs = self.buff["o_p"][0].shape[0]

        seqs: Dict[str, List[torch.Tensor]] = {k: [] for k in self.buff.keys()}
        for _ in range(batch_size):
            # sample env index and start time
            env_idx = torch.randint(0, num_envs, (1,)).item()
            start = torch.randint(0, S - T + 1, (1,)).item()
            for k in self.buff.keys():
                # collect T steps for this env and stack -> [T, dim]
                stacked = torch.stack(self.buff[k][start : start + T], dim=0)  # [T, num_envs, dim]
                seq = stacked[:, env_idx, ...]  # [T, dim]
                seqs[k].append(seq)

        # Convert lists to tensors: [T, B, dim]
        out = {}
        for k, v in seqs.items():
            out[k] = torch.stack(v, dim=1)
        return out


def load_runner_policy_from_ckpt(env_wrapper: RslRlVecEnvWrapper, agent_cfg: object, ckpt_path: str, device: str):
    """Instantiate an OnPolicyRunnerWild and load checkpoint to obtain an inference policy.

    Returns: callable policy that accepts a TensorDict and returns actions tensor.
    """
    agent_dict = asdict(agent_cfg)
    runner = OnPolicyRunnerWild(env_wrapper, agent_dict, log_dir=None, device=device)
    runner.load(ckpt_path, map_location=device)
    policy = runner.get_inference_policy(device=device)
    return policy, runner


def collect_teacher_rollouts(env_wrapper: RslRlVecEnvWrapper, policy_callable, buffer: ReplayBuffer, cfg: StudentTrainerCfg):
    """Rollout env with the teacher policy and store observations and actions.

    `policy_callable` should accept a TensorDict (as returned by
    `env_wrapper.get_observations()`) and return an actions tensor shaped
    (num_envs, action_dim).
    """
    # reset env
    obs_td, extras = env_wrapper.reset()
    steps = 0
    while steps < cfg.num_steps_collect:
        # get current observations (TensorDict)
        obs_td = env_wrapper.get_observations()
        # call teacher policy (returns Tensor or TensorDict depending on implementation)
        with torch.inference_mode():
            actions = policy_callable(obs_td.to(cfg.device))
        # step env
        next_obs_td, rewards, dones, extras = env_wrapper.step(actions.to(env_wrapper.device))

        # Extract groups: policy (proprio), extero, critic (privileged)
        # Convert TensorDict entries to tensors on CPU
        o_p = obs_td["policy"].clone().cpu()
        o_e = obs_td["extero"].clone().cpu()
        # privileged/critic group
        s_p = obs_td["critic"].clone().cpu()
        # actions may be tensor or TensorDict
        if hasattr(actions, "to"):
            act_cpu = actions.clone().cpu()
        else:
            act_cpu = torch.tensor(actions)

        buffer.add_step(o_p, o_e, act_cpu, s_p)
        steps += o_p.shape[0]
        # handle dones: env_wrapper.reset() is called internally by wrapper if needed
    return buffer


def build_student(args: Any, model_cfg: Dict) -> nn.Module:
    """Instantiate student policy (belief encoder + policy head).

    The repo provides `RecurrentAttentionPolicy` which can be used for the
    student. The caller must prepare `args` and `model_cfg` consistent with the
    training environment.
    """
    student = RecurrentAttentionPolicy(args, model_cfg)
    return student


def train_student_loop(student: nn.Module, buffer: ReplayBuffer, cfg: StudentTrainerCfg):
    optim = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=1e-5)

    for epoch in range(cfg.max_epochs):
        if buffer.size() < cfg.seq_len:
            print("Not enough data yet; collecting more")
            break
        batch = buffer.sample_sequences(cfg.batch_size)
        # batch: dict of [T, B, ...]
        o_p = batch["o_p"]
        o_e = batch["o_e"]
        teacher_actions = batch["action"]
        s_p = batch["s_p"]

        # Forward through student (use_decoder=True during training)
        out = student(o_p, o_e, hidden_state=None, use_decoder=True)
        student_actions = out["action"]  # [T, B, action_dim]
        estimated_extero = out.get("estimated_extero_state")

        # Behavior cloning loss (MSE)
        l_behavior = nn.MSELoss()(student_actions, teacher_actions)
        # Reconstruction loss (reconstruction of privileged state)
        # NOTE: This scaffold assumes decoder reconstructs extero. Replace with
        # a dedicated privileged-state decoder if available.
        if estimated_extero is not None:
            l_rec = nn.MSELoss()(estimated_extero, s_p)
        else:
            l_rec = torch.tensor(0.0)

        loss = 1.0 * l_behavior + 0.5 * l_rec

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optim.step()

        print(f"Epoch {epoch}: loss={loss.item():.6f} bc={l_behavior.item():.6f} rec={l_rec.item():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=str, nargs="?", default=None)
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.task_id is None:
        tasks = list_tasks()
        if not tasks:
            raise RuntimeError("No tasks available in mjlab.tasks")
        task_id = tasks[0]
        print(f"No task_id given, using first task: {task_id}")
    else:
        task_id = args.task_id

    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env_wrapper = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    cfg = StudentTrainerCfg(task_id=task_id, teacher_ckpt=args.teacher_ckpt, device=args.device)

    # Buffer
    buffer = ReplayBuffer(seq_len=cfg.seq_len)

    # Load teacher policy via runner
    policy_callable, runner = load_runner_policy_from_ckpt(env_wrapper, agent_cfg, cfg.teacher_ckpt, cfg.device)

    # Collect rollouts (this stores per-step tensors into buffer)
    collect_teacher_rollouts(env_wrapper, policy_callable, buffer, cfg)

    # Save collected dataset for offline training or debugging
    out_dir = Path("datasets")
    out_dir.mkdir(exist_ok=True)
    torch.save(buffer.buff, out_dir / f"student_buffer_{task_id}.pt")
    print(f"Saved collected dataset to {out_dir / f'student_buffer_{task_id}.pt'}")

    # NOTE: next steps: instantiate student with proper args (observation dims)
    # and call `train_student_loop(student, buffer, cfg)`.


if __name__ == "__main__":
    main()
