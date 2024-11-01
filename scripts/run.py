import sys 
import numpy as np
import torch
from madrona_simple_example import GridWorld
import pygame
import os
import csv
import sys
import argparse
import time

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--visualize', action='store_true', help="Enable visualization")
arg_parser.add_argument('--logs', action='store_true', help="Enable logging")

arg_parser.add_argument('--num_worlds', type=int, default=1)
arg_parser.add_argument('--num_steps', type=int, default=5)

arg_parser.add_argument('--use_gpu', type=bool, default=False)
arg_parser.add_argument('--pos_logs_path', type=str, default="pos_logs.bin")
args = arg_parser.parse_args()

PLAYER_CIRCLE_SIZE = 20  # Circle size, representing players
SCREEN_WIDTH, SCREEN_HEIGHT = 940, 500 

# Init
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Basketball Court Simulation")

# Load court image
court_img = pygame.image.load("court.png")
court_img = pygame.transform.scale(court_img, (SCREEN_WIDTH, SCREEN_HEIGHT))

num_worlds = args.num_worlds

enable_gpu_sim = False
if args.use_gpu:
    enable_gpu_sim = True

array_shape = [5,6]
walls = np.zeros(array_shape)
rewards = np.zeros(array_shape)
walls[3, 2:] = 1
start_cell = np.array([4,5])
end_cell = np.array([[4,5]])
rewards[4, 0] = -1
rewards[4, 5] = 1

# Creating array of player positions
num_players = 10
players = []

points = []

# initial player positions
for i in range(num_players):
    # x = np.random.uniform(0, 94)
    # y = np.random.uniform(0, 50)
    points.append([i * 5, i * 5])
print(points)


# Create simulator object (need to rename)
grid_world = GridWorld(points, num_worlds, start_cell, end_cell, rewards, walls, enable_gpu_sim, 0)
#grid_world.vis_world()

print(grid_world.observations.shape)

if args.logs:
    # Creates file if doesn't exist
    if not os.path.exists(args.pos_logs_path):
        open(args.pos_logs_path, 'wb').close()

import torch

# Asumming tensor is (num_worlds, 2 * num_players)
# Modified Jermaine's Code
def load_agents_from_tensor(tensor):
    worlds_agents = []
    
    for world_index, world_data in enumerate(tensor):
        agents = []
        
        for player_id in range(num_players): 
            agent_id = player_id
            # Extract x and y for the player
            x = float(world_data[player_id][0]) * 5
            y = float(world_data[player_id][1]) * 5
            
            # Assign color based on player ID
            if 0 <= agent_id <= 4:
                color = "#002B5C"  # Blue
            elif 5 <= agent_id <= 9:
                color = (255, 170, 51)  # Yellow
            else:
                color = "#ff8c00"  # Basketball
            
            # Create agent dictionary
            agent = {
                'id': agent_id + 1,  # Increment ID by 1
                'x': x,
                'y': y,
                'color': color
            }
            agents.append(agent)
        
        # Append the agents for this world to the main list
        worlds_agents.append({'world_index': world_index, 'agents': agents})
    
    return worlds_agents

def draw_agents(screen, world):
    agents = world[0]['agents']

    for agent in agents:
        # Using center circle as (0,0)
        screen_x = SCREEN_WIDTH / 2 + agent['x']
        screen_y = SCREEN_HEIGHT / 2 - agent['y']  # Y axis is opposite

        pygame.draw.circle(screen, agent['color'], (int(screen_x), int(screen_y)), PLAYER_CIRCLE_SIZE)
        
        # Showing ID
        font = pygame.font.SysFont(None, 24)
        text = font.render(str(agent['id']), True, (255, 255, 255))
        screen.blit(text, (int(screen_x) - 10, int(screen_y) - 10))

# Right now, the code simply increments player position by 1 each loop
# the Player positions tensor is of shape, (num_worlds, num_players * 2)
# Where each pair of 2 elements (ex. index 2 and index 3) correspond to the x and y position of a player
# Its a hacked together solution, but hopefully the meeting tommorow can help with that. 
for i in range(args.num_steps):
    # Advance simulation across all worlds
    grid_world.step()

    if args.logs and not args.visualize:
        with open(args.pos_logs_path, 'ab') as pos_logs:
            pos_logs.write(grid_world.player_pos.numpy().tobytes())

    elif args.logs and args.visualize:
        agents = load_agents_from_tensor(grid_world.player_pos)  

        screen.blit(court_img, (0, 0)) 
        draw_agents(screen, agents) 
        pygame.display.flip() 
        time.sleep(1)

pygame.quit()
sys.exit()

def load_agents_from_csv(file_path):
    agents = []
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            agent_id = int(row['AgentID'])

            if 0 <= agent_id <= 4:
                color = "#002B5C"  # Blue
            elif 5 <= agent_id <= 9:
                color = (255,170,51)  # Yellow
            else:
                color = "#ff8c00"  # Basketball

            agent = {
                'id': int(row['AgentID'])+1,
                'x': float(row['GlobalX'])*40,    # *40 in order to amplify the position interval, change in future
                'y': float(row['GlobalY'])*40,
                'color': color
            }
            agents.append(agent)
    return agents