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
import json
import random
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from multi_agent_train import BasketballMultiAgentEnv 
from ray.rllib.policy.policy import Policy


PLAYERS_PER_TEAM = 2

LEFT_HOOP_X = -41.75
RIGHT_HOOP_X = 41.75
PASSING_VELOCITY = 30.0

class SimulationPolicies:
    def __init__(self, debug_mode_on=False):
        self.debug_mode_on = debug_mode_on

    def setDebugMode(self, new_debug_mode):
        self.debug_mode_on = new_debug_mode
        print(self.debug_mode_on, " what is this val")

    # Takes in a function (most often print)
    # Only executes the function if debug mode is on
    def print(self, *x):
        if self.debug_mode_on:
            for elem in x:
                print(elem, end=" ")
            print("\n")

    def goto_position(self, world_index, agent_index, goal_position, desired_velocity):
        # Get the agent's current position and facing angle
        x = self.grid_world.player_pos[0][agent_index][0]
        y = self.grid_world.player_pos[0][agent_index][1]
        v = self.grid_world.player_pos[0][agent_index][3]
        facing_angle = self.grid_world.player_pos[0][agent_index][5]

        # Compute the vector towards the goal
        dx = goal_position[0] - x
        dy = goal_position[1] - y

        # Compute the angle towards the goal
        desired_direction = np.arctan2(dy, dx)

        # Stop early if reaches goal already
        if 2*v/8 < np.hypot(abs(dx),abs(dy)) < 3*v/8 :
            self.print("Yes, reached!")
            self.grid_world.actions[world_index, agent_index, 0] = 0.0
            self.grid_world.actions[world_index, agent_index, 1] = 0.0
            self.grid_world.actions[world_index, agent_index, 2] = 0.0
            return True

        # Normalize desired_direction to be between -pi and pi
        desired_direction = (desired_direction + np.pi) % (2 * np.pi) + np.pi

        # Compute the difference between current facing angle and desired direction
        angle_diff = desired_direction - facing_angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # Set angular velocity proportional to angle difference
        angular_velocity = angle_diff * 2.0  # Scaling factor

        self.grid_world.actions[world_index, agent_index, 0] = desired_velocity
        self.grid_world.actions[world_index, agent_index, 1] = desired_direction
        self.grid_world.actions[world_index, agent_index, 2] = angular_velocity

        return False

    def initialize_run_in_line(self):
        # Initialize agent states
        agents_state = [{'state': 'waiting', 'timer': 0.0} for _ in range(self.num_players)]
        spacing = 5.0  # No interval between players
        line_start_x = -28.0  # Starting position at the free throw line (-10, 0)
        line_y = 0.0

        for i in range(self.num_players):
            x = line_start_x + spacing * i
            y = line_y
            self.grid_world.player_pos[0][i][0] = x  # x position
            self.grid_world.player_pos[0][i][1] = y  # y position
            self.grid_world.player_pos[0][i][5] = 0.0  # facing angle

        # Set first player to 'at_free_throw_line'
        agents_state[0]['state'] = 'at_free_throw_line'
        agents_state[0]['timer'] = 0.0
        return agents_state
    
    
    def run_in_line_policy(self, agents_state):
        spacing = 5.0  # No interval between players
        line_start_x = -28.0  # Starting position at the free throw line (-10, 0)
        for j in range(self.num_worlds):
            for agent_index in range(self.num_players):
                state = agents_state[agent_index]['state']
                timer = agents_state[agent_index]['timer']

                if state == 'waiting':
                    # The agent is in line, remain stationary
                    self.grid_world.actions[j, agent_index] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0,0])

                elif state == 'at_free_throw_line':
                    # The agent is at the free throw line, wait for 1 second
                    agents_state[agent_index]['timer'] += self.dt
                    self.grid_world.actions[j, agent_index] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

                    if agents_state[agent_index]['timer'] >= 1.0:
                        # After 1 second, change state to 'going_up'
                        agents_state[agent_index]['state'] = 'going_up'

                elif state == 'going_up':
                    goal_position = (0.0, 20.0)
                    desired_velocity = 30.0
                    if self.goto_position(j, agent_index, goal_position, desired_velocity):
                        agents_state[agent_index]['state'] = 'returning_to_line'

                elif state == 'returning_to_line':
                    goal_x = line_start_x + spacing * (self.num_players - 1) + spacing
                    goal_position = (goal_x, 0.625)
                    desired_velocity = 30.0
                    if self.goto_position(j, agent_index, goal_position, desired_velocity):
                        agents_state[agent_index]['state'] = 'waiting'
                        agents_state[agent_index]['completed'] = True
                        agents_state[agent_index]['needs_scoot'] = True

                for agent_index in range(self.num_players):
                    if agents_state[agent_index].get('needs_scoot', False):
                        scoot_distance = 5.0
                        for i in range(self.num_players):
                            x = self.grid_world.player_pos[0][i][0]
                            self.grid_world.player_pos[0][i][0] = x - scoot_distance

                        next_player_index = (agent_index + 1) % self.num_players
                        agents_state[next_player_index]['state'] = 'at_free_throw_line'
                        agents_state[next_player_index]['timer'] = 0.0
                        agents_state[agent_index]['needs_scoot'] = False
        return agents_state
    
    def make_pass(self, world_index, agent_index, pass_position, pass_velocity):
        x = self.grid_world.player_pos[world_index][agent_index][0]
        y = self.grid_world.player_pos[world_index][agent_index][1]

        dx = pass_position[0] - x
        dy = pass_position[1] - y

        desired_direction = (np.arctan2(dy, dx) + np.pi) % (2 * np.pi) + np.pi

        self.grid_world.actions[world_index][agent_index][3] = desired_direction
        self.grid_world.actions[world_index][agent_index][4] = pass_velocity

    def different_goto_position(self, world_index, agent_index, goal_position, desired_velocity):
        # Get the agent's current position and facing angle
        x = self.grid_world.player_pos[0][agent_index][0]
        y = self.grid_world.player_pos[0][agent_index][1]
        v = self.grid_world.player_pos[0][agent_index][3].item()
        facing_angle = self.grid_world.player_pos[0][agent_index][5]

        # Compute the vector towards the goal
        dx = goal_position[0] - x
        dy = goal_position[1] - y

        # Compute the angle towards the goal
        desired_direction = np.arctan2(dy, dx)

        # Stop early if reaches goal already
        if (np.hypot(abs(dx),abs(dy)) < 0.25):
            self.grid_world.actions[world_index, agent_index, 0] = 0.0
            self.grid_world.actions[world_index, agent_index, 1] = 0.0
            self.grid_world.actions[world_index, agent_index, 2] = 0.0
            return True

        # Normalize desired_direction to be between -pi and pi
        desired_direction = (desired_direction + np.pi) % (2 * np.pi) + np.pi

        # Compute the difference between current facing angle and desired direction
        angle_diff = desired_direction - facing_angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        desired_velocity = min(desired_velocity, 10.0 * np.hypot(abs(dx),abs(dy)))
        # Set angular velocity proportional to angle difference
        angular_velocity = angle_diff * 2.0  # Scaling factor
        self.grid_world.actions[world_index, agent_index, 0] = desired_velocity
        self.grid_world.actions[world_index, agent_index, 1] = desired_direction
        self.grid_world.actions[world_index, agent_index, 2] = angular_velocity

        return False
        
    def get_velocity_angle_for_ball_pass(self, world_index, agent_index, desired_velocity):
        if agent_index == -1:
            return
        if agent_index < PLAYERS_PER_TEAM:
            # Return a random number between 0 and 4 that is not agent_index
            possible_indices = [i for i in range(PLAYERS_PER_TEAM) if i != agent_index]
        else:
            # Return a random number between PLAYERS_PER_TEAM and 9 that is not agent_index
            possible_indices = [i for i in range(PLAYERS_PER_TEAM, PLAYERS_PER_TEAM * 2) if i != agent_index]
    
        target_agent_index = random.choice(possible_indices)
        self.print("possible_indices:", possible_indices, "\t agent_index:", agent_index, "\t target_agent_index:", target_agent_index)

        x, y = self.grid_world.player_pos[0][target_agent_index][0], self.grid_world.player_pos[0][target_agent_index][1]
        self.make_pass(world_index, agent_index, (x, y), desired_velocity)

    def run_around_and_defend_initialize(self):
        # Initialize agent states
        agents_state = []
        team_holding = self.grid_world.who_holds[self.current_viewed_world][0].item() // PLAYERS_PER_TEAM

        for i in range(self.num_players):
            self.grid_world.choices[self.current_viewed_world][i] = 0
            if (i // PLAYERS_PER_TEAM) ==  team_holding:
                agents_state.append({'state': 'running'})
            else:
                agents_state.append({'state': 'defending'})
        self.print('initialized')
        
        return agents_state
    
    def defend_player(self, cur_player, mark_player):
        if (cur_player // PLAYERS_PER_TEAM) == 0:
            hoop = RIGHT_HOOP_X
        else:
            hoop = LEFT_HOOP_X
        x = self.grid_world.player_pos[self.current_viewed_world][mark_player][0].item() * 0.75 + hoop * 0.25
        y = self.grid_world.player_pos[self.current_viewed_world][mark_player][1].item() * 0.8

        x = x * 0.95 + self.grid_world.ball_pos[self.current_viewed_world][0].item() * 0.05
        y = y * 0.95 + self.grid_world.ball_pos[self.current_viewed_world][1].item() * 0.05
        gpos = (x, y)
        self.different_goto_position(self.current_viewed_world, cur_player, gpos, 20.0)


    def run_around_and_defend_policy(self, agents_state):
        for j in range(self.num_worlds):
            for agent_index in range(self.num_players):
                if (self.grid_world.who_holds[self.current_viewed_world][0].item() != -1):
                    if agents_state[agent_index]['state'] == 'passing':
                        pass
                    elif (self.grid_world.who_holds[self.current_viewed_world][0].item() // PLAYERS_PER_TEAM == agent_index // PLAYERS_PER_TEAM):
                        agents_state[agent_index]['state'] = 'running'
                    else:
                        agents_state[agent_index]['state'] = 'defending'
                state = agents_state[agent_index]['state']
                if ((self.grid_world.who_holds[self.current_viewed_world][1].item() == -1)
                    and self.grid_world.who_holds[self.current_viewed_world][0].item() == -1):
                    self.different_goto_position(self.current_viewed_world, 
                                                 agent_index, 
                                                 (self.grid_world.ball_pos[self.current_viewed_world][0].item(),
                                                  self.grid_world.ball_pos[self.current_viewed_world][1].item()), 
                                                 20.0)
                elif state == 'running':
                    ypos = (int(self.elapsed_time) // 6) % 2
                    xpos = (agent_index // PLAYERS_PER_TEAM)
                    if (agent_index % PLAYERS_PER_TEAM == 0):
                        gpos = (-10.0 + xpos * 20.0, -20.0 + ypos * 35.0)
                    elif (agent_index % PLAYERS_PER_TEAM == 1):
                        gpos = (-41.0 + xpos * 82.0, -21.0 + ypos * 14.0)
                    elif (agent_index % PLAYERS_PER_TEAM == 2):
                        gpos = (-30.0 + xpos * 60.0, 8.0 + ypos * 13.0)
                    elif (agent_index % PLAYERS_PER_TEAM == 3):
                        gpos = (-28.0 + xpos * 56.0, -16.0 + ypos * 18.0)
                    elif (agent_index % PLAYERS_PER_TEAM == 4):
                        gpos = (-40.0 + xpos * 80.0, ypos * 18.0)
                    self.different_goto_position(self.current_viewed_world, agent_index, gpos, 20.0)
                elif (state == 'defending'):
                    self.defend_player(agent_index, (agent_index + PLAYERS_PER_TEAM) % (PLAYERS_PER_TEAM * 2) )
                elif (state == 'passing'):
                    self.get_velocity_angle_for_ball_pass(self.current_viewed_world, agent_index, PASSING_VELOCITY)
                    self.print("PASSING: \n\n ", self.grid_world.actions[self.current_viewed_world, agent_index])
        return agents_state

    
    def do_nothing_i(self):
        return {}
    
    def do_nothing(self, agents_state):
        return {}
    

    def initialize_PPO(self):
        
        offense_policy = Policy.from_checkpoint(r"C:\Users\travi\repos\madrona_simple_example\scripts\checkpoints\evenbettermodels\iter_950\policies\offense")
        defense_policy = Policy.from_checkpoint(r"C:\Users\travi\repos\madrona_simple_example\scripts\checkpoints\evenbettermodels\iter_950\policies\defense")
        return offense_policy, defense_policy

    def get_PPO_actions(self, offense_policy, defense_policy):
        shared_obs = {
            "player_pos": np.float32(self.grid_world.player_pos[0].numpy()).flatten(),
            "ball_pos": np.float32(self.grid_world.ball_pos[0].numpy()),
            "who_holds": int(np.add(self.grid_world.who_holds[0][0].numpy(), [1])[0]),
            "who_shot": int(np.add(self.grid_world.who_holds[0][1].numpy(), [1])[0]),
            "who_passed": int(np.add(self.grid_world.who_holds[0][2].numpy(), [1])[0]),
            "ball_state": int(self.grid_world.who_holds[0].numpy()[3]),
            "scoreboard": self.grid_world.scoreboard[0].numpy()
        }
        
        one_hot_sizes = {
            "who_holds": 5,
            "who_shot": 5,
            "who_passed": 5,
            "ball_state": 6
        }

        final_obs_list = []

        sorted_keys = sorted(shared_obs.keys())

        for key in sorted_keys:
            value = shared_obs[key]
            if key in one_hot_sizes:  
                one_hot_vector = np.eye(one_hot_sizes[key], dtype=np.float32)[value]
                final_obs_list.append(one_hot_vector)
            else: 
                final_obs_list.append(value)

        final_obs = np.concatenate(final_obs_list)
        offense_action = offense_policy.compute_single_action(obs=final_obs)[0]
        defense_action = defense_policy.compute_single_action(obs=final_obs)[0]

        off_act1 = np.clip(offense_action["player1"], -1, 1)
        off_act2 = np.clip(offense_action["player2"], -1, 1)

        def_act1 = np.clip(defense_action["player1"], -1, 1)
        def_act2 = np.clip(defense_action["player2"], -1, 1)

        off_act1[0] += 1
        off_act2[0] += 1
        def_act1[0] += 1
        def_act2[0] += 1

        off_act1[0] *= 15.0
        off_act2[0] *= 15.0
        def_act1[0] *= 15.0
        def_act2[0] *= 15.0

        off_act1[1] *= np.pi
        off_act2[1] *= np.pi
        def_act1[1] *= np.pi
        def_act2[1] *= np.pi

        off_act1[3] *= np.pi
        off_act2[3] *= np.pi
        def_act1[3] *= np.pi
        def_act2[3] *= np.pi

        off_act1[4] += 1
        off_act2[4] += 1
        def_act1[4] += 1
        def_act2[4] += 1

        off_act1[4] *= 25.0
        off_act2[4] *= 25.0
        def_act1[4] *= 25.0
        def_act2[4] *= 25.0

        self.grid_world.actions[0][0] = torch.tensor(off_act1)
        self.grid_world.actions[0][1] = torch.tensor(off_act2)
        self.grid_world.actions[0][2] = torch.tensor(def_act1)
        self.grid_world.actions[0][3] = torch.tensor(def_act2)
        self.grid_world.choices[0] = torch.tensor(
            [offense_action["decision"]] * 2 + [0] * 2
        ).view(4, 1)