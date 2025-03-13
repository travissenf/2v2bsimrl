import sys
import numpy as np
import torch
from madrona_simple_example import GridWorld
import pygame
import os
import csv
import argparse
import time
from datetime import datetime
from moviepy.editor import ImageSequenceClip
import cv2
import tkinter as tk
from tkinter import messagebox
from policies import SimulationPolicies
import json
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.tune.registry import register_env
import gymnasium
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility

P_LOC_INDEX_TO_VAL = {0: "x", 1: "y", 2: "theta", 3: "velocity", 4:"angular v", 5: "facing angle"}
P_LOC_VAL_TO_INDEX = {"x": 0, "y": 1, "theta": 2, "velocity": 3, "angular v": 4, "facing angle": 5}

B_LOC_INDEX_TO_VAL = {0: "x", 1: "y", 2: "theta", 3: "velocity"}

SUPPORTED_POLICIES = {'run_in_line', 'run_and_defend', 'do_nothing'}
PASSING_VELOCITY = 35.0


class BasketballMultiAgentEnv(MultiAgentEnv):
    def __init__(self, grid_world):
        super().__init__()
        self.grid_world = grid_world
        self.agents = ["offense", "defense"]
        self.reset_path = "gamestates/2v2init.json"

        offense_action_space = spaces.Dict({
            "player1": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "player2": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "decision": spaces.Discrete(3)
        })
        
        defense_action_space = spaces.Dict({
            "player1": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "player2": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "decision": spaces.Discrete(3)
        })
        
        offense_obs_space = spaces.Dict({
            "player_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4, 6), dtype=np.float32),
            "ball_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            "who_holds": spaces.Discrete(5),
            "who_shot": spaces.Discrete(5),
            "who_passed": spaces.Discrete(5),
            "ball_state": spaces.Discrete(6),
            "scoreboard": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.int32)
        })
        
        defense_obs_space = spaces.Dict({
            "player_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4, 6), dtype=np.float32),
            "ball_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            "who_holds": spaces.Discrete(5),
            "who_shot": spaces.Discrete(5),
            "who_passed": spaces.Discrete(5),
            "ball_state": spaces.Discrete(6),
            "scoreboard": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.int32)
        })
        
        # RLlib requires these specific attributes with the proper spaces
        self.observation_space = spaces.Dict({
            "offense": offense_obs_space,
            "defense": defense_obs_space
        })
        
        self.action_space = spaces.Dict({
            "offense": offense_action_space,
            "defense": defense_action_space
        })

    def get_action_space(self, agent_id):
        return self.action_space[agent_id]
    
    def get_observation_space(self, agent_id):
        return self.observation_space[agent_id]


    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.grid_world.reset(self.reset_path)
        obs = self._get_obs()
        infos = {agent_id: {} for agent_id in obs.keys()}
        return obs, infos

    def step(self, action_dict):
        offense_action = action_dict["offense"]
        defense_action = action_dict["defense"]
        
        self.grid_world.actions[0][0] = torch.tensor(offense_action["player1"])
        self.grid_world.actions[0][1] = torch.tensor(offense_action["player2"])
        self.grid_world.actions[0][2] = torch.tensor(defense_action["player1"])
        self.grid_world.actions[0][3] = torch.tensor(defense_action["player2"])
        self.grid_world.choices[0] = torch.tensor([offense_action["decision"]] * 4)
        self.grid_world.step()

        obs = self._get_obs()
        rewards = self._compute_rewards()
        
        # Determine if episode is done due to actual completion (terminated)
        # or due to time limits (truncated)
        is_terminated = (
            self.grid_world.scoreboard[0][0].item() != 0 or
            abs(self.grid_world.ball_pos[0][0].item()) > 47.0 or
            abs(self.grid_world.ball_pos[0][1].item()) > 25.0 or
            self.grid_world.who_holds[0][0].item() > 1 or
            self.grid_world.foul_call[0].numpy().sum() != 0
        )
        
        # Check if truncated due to time limit (shot clock)
        is_truncated = self.grid_world.scoreboard[0][3].item() >= 24 / 0.05
        
        # Create per-agent dictionaries for terminated and truncated states
        terminateds = {"offense": is_terminated, "defense": is_terminated, "__all__": is_terminated}
        truncateds = {"offense": is_truncated, "defense": is_truncated, "__all__": is_truncated}
        
        # Create empty info dictionary per agent
        infos = {"offense": {}, "defense": {}}
        
        return obs, rewards, terminateds, truncateds, infos
    
    def _get_obs(self):
        print("cock")
        print(type(self.grid_world.who_holds[0][0].item()[0] + 1))
        return {
            "offense": {
                "player_pos": self.grid_world.player_pos[0].numpy(),
                "ball_pos": self.grid_world.ball_pos[0].numpy(),
                "who_holds": self.grid_world.who_holds[0][0].item()[0] + 1,
                "who_shot": self.grid_world.who_holds[0][1][0].item() + 1,
                "who_passed": self.grid_world.who_holds[0][2][0].item() + 1,
                "ball_state": self.grid_world.who_holds[0][3][0].item(),
                "scoreboard": self.grid_world.scoreboard[0].numpy()
            },
            "defense": {
                "player_pos": self.grid_world.player_pos[0].numpy(),
                "ball_pos": self.grid_world.ball_pos[0].numpy(),
                "who_holds": self.grid_world.who_holds[0][0].item()[0] + 1,
                "who_shot": self.grid_world.who_holds[0][1][0].item() + 1,
                "who_passed": self.grid_world.who_holds[0][2][0].item() + 1,
                "ball_state": self.grid_world.who_holds[0][3][0].item(),
                "scoreboard": self.grid_world.scoreboard[0].numpy()
            }
        }
    
    def _compute_rewards(self):
        reward_offense = 0
        reward_defense = 0
        
        if self.grid_world.scoreboard[0][0].item() != 0:
            reward_offense += self.grid_world.scoreboard[0][0].item() * 5
            reward_defense -= self.grid_world.scoreboard[0][0].item() * 5
        
        if abs(self.grid_world.ball_pos[0][0]) > 47.0 or abs(self.grid_world.ball_pos[0][1]) > 25.0 :
            reward_offense -= 3
            reward_defense += 3
        
        if self.grid_world.who_holds[0][0] > 1:
            reward_defense += 3
            reward_offense -= 3
        
        if self.grid_world.foul_call[0][0].item() != 0 or self.grid_world.foul_call[0][1].item() != 0:
            reward_defense += 5
            reward_offense -= 5

        if self.grid_world.foul_call[0][2].item() != 0 or self.grid_world.foul_call[0][3].item() != 0:
            reward_defense -= 4
            reward_offense += 4
        
        if self.grid_world.scoreboard[0][3].item():
            reward_defense += 3
            reward_offense -= 3
        return {
            "offense": reward_offense,
            "defense": reward_defense
        }

if __name__ == "__main__":
    ray.init()

    env_name = "basketball_env"

    def env_creator(env_config):
        points = []
        for i in range(4):
            points.append([(i - 5) * 5, (i - 5) * 5, 0, 0.0, 0.0, -np.pi])

        grid_world = GridWorld(points, 1, False, 0)  
        grid_world.reset("gamestates/2v2init.json")
        return MultiAgentEnvCompatibility(BasketballMultiAgentEnv(grid_world))

    register_env(env_name, env_creator)

    config = (
        PPOConfig()
        .environment(env=env_name, env_config={})
        .multi_agent(
            policies={
                "offense": (None, 
                    spaces.Dict({
                        "player_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4, 6), dtype=np.float32),
                        "ball_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                        "who_holds": spaces.MultiDiscrete([5, 5, 5, 6], start=[-1, -1, -1, 0]),
                        "scoreboard": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.int32)
                    }), 
                    spaces.Dict({
                        "player1": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
                        "player2": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
                        "decision": spaces.Discrete(3)
                    }), 
                    {}
                ),
                "defense": (None, 
                    spaces.Dict({
                        "player_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4, 6), dtype=np.float32),
                        "ball_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                        "who_holds": spaces.MultiDiscrete([5, 5, 5, 6], start=[-1, -1, -1, 0]),
                        "scoreboard": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.int32)
                    }), 
                    spaces.Dict({
                        "player1": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
                        "player2": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
                    }), 
                    {}
                )
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
        )
        .framework("torch")
        .training(train_batch_size=128)  # Smaller batch size to start with
        .evaluation(evaluation_interval=10)  # Evaluate every 10 training iterations
    )
    print('Building Trainer')
    trainer = config.build()
    print('Starting Training')
    for i in range(100):
        result = trainer.train()
        print(f"Iteration {i}: episode_reward_mean={result['episode_reward_mean']}, episode_len_mean={result['episode_len_mean']}")
        if i % 10 == 0:
            checkpoint_dir = trainer.save()
            print(f"Checkpoint saved at {checkpoint_dir}")

    ray.shutdown()
