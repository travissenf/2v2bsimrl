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


PLAYERS_PER_TEAM = 5

LEFT_HOOP_X = -41.75
RIGHT_HOOP_X = 41.75

class SimulationPolicies:
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
            print("Yes, reached!")
            self.grid_world.actions[world_index, agent_index] = torch.tensor([0, 0, 0])
            return True

        # Normalize desired_direction to be between -pi and pi
        desired_direction = (desired_direction + np.pi) % (2 * np.pi) + np.pi

        # Compute the difference between current facing angle and desired direction
        angle_diff = desired_direction - facing_angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # Set angular velocity proportional to angle difference
        angular_velocity = angle_diff * 2.0  # Scaling factor

        self.grid_world.actions[world_index, agent_index] = torch.tensor([desired_velocity, desired_direction, angular_velocity])

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
                    self.grid_world.actions[0, agent_index] = torch.tensor([0.0, 0.0, 0.0])

                elif state == 'at_free_throw_line':
                    # The agent is at the free throw line, wait for 1 second
                    agents_state[agent_index]['timer'] += self.dt
                    self.grid_world.actions[0, agent_index] = torch.tensor([0.0, 0.0, 0.0])

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
        if ((v**2 / (2 * max(np.hypot(abs(dx),abs(dy)), 1e-6))) > 10.0):
            self.grid_world.actions[world_index, agent_index] = torch.tensor([0, 0, 0])
            return True
        if (np.hypot(abs(dx),abs(dy)) < 0.25):
            self.grid_world.actions[world_index, agent_index] = torch.tensor([0, 0, 0])
            return True

        # Normalize desired_direction to be between -pi and pi
        desired_direction = (desired_direction + np.pi) % (2 * np.pi) + np.pi

        # Compute the difference between current facing angle and desired direction
        angle_diff = desired_direction - facing_angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # Set angular velocity proportional to angle difference
        angular_velocity = angle_diff * 2.0  # Scaling factor
        self.grid_world.actions[world_index, agent_index] = torch.tensor([desired_velocity, desired_direction, angular_velocity])

        return False
        
    def get_velocity_angle_for_ball_pass(self, world_index, agent_index, desired_velocity):
        if agent_index < 5:
            # Return a random number between 0 and 4 that is not agent_index
            possible_indices = [i for i in range(5) if i != agent_index]
        else:
            # Return a random number between 5 and 9 that is not agent_index
            possible_indices = [i for i in range(5, 10) if i != agent_index]
    
        target_agent_index = random.choice(possible_indices)

        x, y = self.grid_world.player_pos[0][target_agent_index][0], self.grid_world.player_pos[0][target_agent_index][1]

        self.different_goto_position(world_index, agent_index, (x, y), desired_velocity)

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
        print('initialized')
        
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
                                                 15.0)
                elif state == 'running':
                    ypos = (int(self.elapsed_time) // 4) % 2
                    xpos = (agent_index // 5)
                    if (agent_index % 5 == 0):
                        gpos = (-10.0 + xpos * 20.0, -20.0 + ypos * 35.0)
                    elif (agent_index % 5 == 1):
                        gpos = (-41.0 + xpos * 82.0, -21.0 + ypos * 14.0)
                    elif (agent_index % 5 == 2):
                        gpos = (-30.0 + xpos * 60.0, 8.0 + ypos * 13.0)
                    elif (agent_index % 5 == 3):
                        gpos = (-28.0 + xpos * 56.0, -16.0 + ypos * 18.0)
                    elif (agent_index % 5 == 4):
                        gpos = (-40.0 + xpos * 80.0, ypos * 18.0)
                    self.different_goto_position(self.current_viewed_world, agent_index, gpos, 15.0)
                elif (state == 'defending'):
                    self.defend_player(agent_index, (agent_index + 5) % 10)
                elif (state == 'passing'):
                    self.get_velocity_angle_for_ball_pass(self.current_viewed_world, agent_index, 40)
                    print("PASSING: \n\n ", self.grid_world.actions[self.current_viewed_world, agent_index])
        return agents_state

