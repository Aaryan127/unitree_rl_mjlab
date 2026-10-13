import argparse
import yaml
import numpy as np
import torch
import os

from mjlab.rsl_rl.modules.main_student import VisionStudentAgent
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.rsl_rl.runners import OnPolicyRunner
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from dataclasses import asdict, dataclass, field
from typing import Literal, cast

# @dataclass(frozen=True)
# class TrainConfig:
#     env: ManagerBasedRlEnvCfg
#     agent: RslRlOnPolicyRunnerCfg
#     motion_file: str | None = None
#     video: bool = False
#     video_length: int = 200
#     video_interval: int = 2000
#     enable_nan_guard: bool = False
#     torchrunx_log_dir: str | None = None
#     wandb_run_path: str | None = None
#     gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

#     @staticmethod
#     def from_task(task_id: str) -> "TrainConfig":
#         env_cfg = load_env_cfg(task_id)
#         agent_cfg = load_rl_cfg(task_id)
#         assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
#         return TrainConfig(env=env_cfg, agent=agent_cfg)

# class Environment:
#     """
#     Change to the environment you are using
#     """
#     def __init__(self, args):
#         self.obs_dim = args.proprio_obs_dim + args.extero_obs_dim
#         self.action_dim = args.action_dim
#         self.n_envs = args.n_envs

#     def observe(self):
#         observations = np.random.normal(size=(self.n_envs, self.obs_dim)).astype(np.float32)
#         return observations

#     def observe_noisy(self):
#         noisy_observations = np.random.normal(size=(self.n_envs, self.obs_dim)).astype(np.float32)
#         return noisy_observations

#     def step(self, action):
#         rewards = np.random.normal(size=self.n_envs).astype(np.float32)
#         dones = np.zeros(shape=self.n_envs).astype(np.bool_)
#         return rewards, dones

class TeacherAgent:
    """
    Change to the teacher agent you are using
    """
    # def __init__(self, args):
    #     self.action_dim = args.action_dim
    #     self.n_envs = args.n_envs

    # def getAction(self, observations):
    #     actions = np.random.normal(size=(self.n_envs, self.action_dim)).astype(np.float32)
    #     return actions

    def __init__(self, checkpoint_path, env, agent_cfg, device):
        self.runner = OnPolicyRunner(env, agent_cfg, log_dir="", device=device)
        self.runner.load(checkpoint_path)

        self.policy = self.runner.alg.policy
        self.policy.eval()
        self.device = device

    def getAction(self, observations):
        obs_tensor = torch.tensor(observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions = self.policy.act(obs_tensor, deterministic=True)
        return actions


def getParser():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--name', type=str, default='example')
    parser.add_argument('--device', type=str, default='cuda', help='gpu or cpu.')
    parser.add_argument('--save_dir', type=str, default='example', help='directory name to save weights')
    return parser

if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    # parameters to be set from the environment you are running
    args.student_model_num = 0
    args.student_lr = 3e-4
    args.student_epochs = 1
    args.max_grad_norm = 1.
    args.student_policy_type = "vision_recurrent"
    args.n_envs = 100
    args.n_steps = 40000
    args.n_steps_per_env = int(args.n_steps / args.n_envs)

    # check from obs space
    # args.proprio_obs_dim = 10 #100
    args.extero_obs_dim = 20 #200
    # args.action_dim = 12

    # args.action_dim = env.action_space.shape[0]
    # args.n_envs = env.num_envs

    # config
    # with open("/media/ok/ubuntu/Quadruped_Projects/legged_rl/unitree_rl_mjlab/mjlab/rsl_rl/modules/config.yaml", "r") as f:
    #     cfg = yaml.safe_load(f)

    # define teacher agent (pretrained)
    # teacher = TeacherAgent(args)
    
    # teacher = TeacherAgent(
    #     checkpoint_path="/media/ok/ubuntu/Quadruped_Projects/legged_rl/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-02-10_22-25-08/model_10000.pt",
    #     env=ManagerBasedRlEnvCfg,
    #     agent_cfg=RslRlOnPolicyRunnerCfg,
    #     device="cuda"
    # )

    task_id = "Mjlab-Velocity-Flat-Unitree-Go2"

    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)

    env=ManagerBasedRlEnv(env_cfg, device="cuda")

    obs_dict, _ = env.reset()

    args.action_dim = env.action_space.shape[0]
    args.n_envs = env.num_envs
    args.proprio_obs_dim= obs_dict["policy"].shape[-1]

    with open("/media/ok/ubuntu/Quadruped_Projects/legged_rl/unitree_rl_mjlab/mjlab/rsl_rl/modules/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    teacher = TeacherAgent(
        checkpoint_path="/media/ok/ubuntu/Quadruped_Projects/legged_rl/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-02-10_22-25-08/model_10000.pt",
        env=env,
        agent_cfg=agent_cfg,
        device="cuda"
    )

    # define student agent
    student = VisionStudentAgent(args, cfg["student_model"])

    # for update in range(max_update):
    #     for _ in range(args.n_steps_per_env):
            
    #         obs = env.observe()

    #         noisy_obs = env.observe_noisy()

    #         proprio_obs = obs[:, :args.proprio_obs_dim]
    #         extero_obs = obs[:, -args.extero_obs_dim:]
    #         noisy_extero_obs = noisy_obs[:, -args.extero_obs_dim:]
    #         proprio_obs_tensor = torch.from_numpy(proprio_obs).to(args.device)
    #         noisy_extero_obs_tensor = torch.from_numpy(noisy_extero_obs).to(args.device)

    #         with torch.no_grad():
    #             # get student action
    #             actions_tensor, hidden_state_tensor = student.getAction(
    #                 proprio_state=proprio_obs_tensor,
    #                 extero_state=noisy_extero_obs_tensor,
    #                 hidden_state=hidden_state_tensor
    #             )

    #             # get teacher action
    #             teacher_actions = teacher.getAction(obs)

    #         actions = actions_tensor.detach().cpu().numpy()
    #         rewards, dones = env.step(actions)

    #         # add data the buffer
    #         student.step(proprio_obs, noisy_extero_obs, extero_obs, teacher_actions)

    #     # train model
    #     loss, reconstruction_loss, action_loss = student.train()

    #     # save model
    #     if update % 5 == 0:
    #         student.save(update)

    #     print('----------------------------------------------------')
    #     print('{:>6}th iteration'.format(update))
    #     print('{:<40} {:>6}'.format("total loss: ", '{:6.4f}'.format(loss)))
    #     print('{:<40} {:>6}'.format("reconstruction loss: ", '{:6.4f}'.format(reconstruction_loss)))
    #     print('{:<40} {:>6}'.format("action loss: ", '{:6.4f}'.format(action_loss)))
    #     print('----------------------------------------------------\n')    

    # obs_dict, _ = env.reset()

    hidden_state = None
    max_update = 10

    for update in range(max_update):

        for _ in range(args.n_steps_per_env):

            # Student sees policy obs
            student_obs = obs_dict["policy"]

            # Teacher sees privileged obs
            teacher_obs = obs_dict["critic"]

            with torch.no_grad():

                student_actions, hidden_state = student.getAction(
                    student_obs,
                    hidden_state
                )

                teacher_actions = teacher.getAction(teacher_obs)

            obs_dict, rewards, terminated, timeouts, _ = env.step(student_actions)

            dones = torch.logical_or(terminated, timeouts)

            if hidden_state is not None:
                hidden_state[:, dones, :] = 0.0

            student.step(student_obs, teacher_actions)

        loss, reconstruction_loss, action_loss = student.train()
        
        if update % 5 == 0:
            student.save(update)

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("total loss: ", '{:6.4f}'.format(loss)))
        print('{:<40} {:>6}'.format("reconstruction loss: ", '{:6.4f}'.format(reconstruction_loss)))
        print('{:<40} {:>6}'.format("action loss: ", '{:6.4f}'.format(action_loss)))
        print('----------------------------------------------------\n')