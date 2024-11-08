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
from datetime import datetime
from moviepy.editor import ImageSequenceClip


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--visualize', action='store_true', help="Enable visualization")
arg_parser.add_argument('--logs', action='store_true', help="Enable logging")

arg_parser.add_argument('--num_worlds', type=int, default=1)
arg_parser.add_argument('--num_steps', type=int, default=1000)

arg_parser.add_argument('--use_gpu', type=bool, default=False)
arg_parser.add_argument('--pos_logs_path', type=str, default="pos_logs.bin")

arg_parser.add_argument('--savevideo', action='store_true', help="Save each frame as an image for video creation")
args = arg_parser.parse_args()

PLAYER_CIRCLE_SIZE = 15  # Circle size, representing players
SCREEN_WIDTH, SCREEN_HEIGHT = 940, 500 

# Init
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Basketball Court Simulation")

# Load agent images
pacman_yellow = pygame.image.load("Pacman_yellow.png")
pacman_blue = pygame.image.load("Pacman_blue.png")
ball_image = pygame.image.load("ball.png")

# Scale images to fit desired player size
pacman_yellow = pygame.transform.scale(pacman_yellow, (PLAYER_CIRCLE_SIZE * 4, PLAYER_CIRCLE_SIZE * 4))
pacman_blue = pygame.transform.scale(pacman_blue, (PLAYER_CIRCLE_SIZE * 4, PLAYER_CIRCLE_SIZE * 4))
ball_image = pygame.transform.scale(ball_image, (PLAYER_CIRCLE_SIZE * 2, PLAYER_CIRCLE_SIZE * 2))



# Load court image
court_img = pygame.image.load("court.png")
court_img = pygame.transform.scale(court_img, (SCREEN_WIDTH, SCREEN_HEIGHT))

num_worlds = args.num_worlds

enable_gpu_sim = False
if args.use_gpu:
    enable_gpu_sim = True

dt = 0.1
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
    # x, y, th, v, angv, facingang
    # points.append([(i-5) * 5, (i-5) * 5, 0, 10.0, 1.0, -np.pi])
    points.append([(i-5) * 5, (i-5) * 5, 0, 0.0, 0.0, -np.pi])

print(points)




# Create simulator object (need to rename)
grid_world = GridWorld(points, num_worlds, start_cell, end_cell, rewards, walls, enable_gpu_sim, 0)
#grid_world.vis_world()

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
        # Load agents data
        agents = []
        
        for player_id in range(num_players): 
            agent_id = player_id
            # Extract x and y for the player
            x = float(world_data[player_id][0]) * 5
            y = float(world_data[player_id][1]) * 5
            th = float(world_data[player_id][2])
            v = float(world_data[player_id][3]) * 5
            facing = float(world_data[player_id][5])
            
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
                'th':th,
                'v':v,
                'facing': facing,
                'color': color
            }
            agents.append(agent)
        
        # Append the agents for this world to the main list
        worlds_agents.append({'world_index': world_index, 'agents': agents})
    
    return worlds_agents

def load_ballpos_from_tensor(tensor):
    worlds_balls = []
    
    for world_index, ball_data in enumerate(tensor):

        x = float(ball_data[0]) *5
        y = float(ball_data[1]) *5
        th = float(ball_data[2])
        v = float(ball_data[3]) *5

        ball = {
            'x': x,
            'y': y,
            'th':th,
            'v':v,
        }

        worlds_balls.append({'world_index': world_index, 'ballpos': ball})
    
    return worlds_balls

def load_whoholds_from_tensor(tensor):
    worlds_whoholds = []
    
    for world_index, whoholds_data in enumerate(tensor):

        whoholds_idx = int(whoholds_data[0])


        whoholds = {
            'whoholds': whoholds_idx
        }

        worlds_whoholds.append({'world_index': world_index, 'whoheld': whoholds})
    
    return worlds_whoholds

def draw_agents(screen, world, world_ball_position):
    agents = world[0]['agents']
    ball_pos = world_ball_position[0]['ballpos']

    for agent in agents:
        # Using center circle as (0,0)
        screen_x = SCREEN_WIDTH / 2 + agent['x']
        screen_y = SCREEN_HEIGHT / 2 - agent['y']  # Y axis is opposite

        # Choose image based on agent ID
        if 0 <= agent['id'] <= 5:
            agent_image = pacman_yellow
        else:
            agent_image = pacman_blue

        # Rotation angle is the player's facing angle
        rotated_image = pygame.transform.rotate(agent_image, np.degrees(agent['facing']))
        rotated_rect = rotated_image.get_rect(center=(screen_x, screen_y))
        screen.blit(rotated_image, rotated_rect.topleft)

        line_length = agent['v']

        movedir = agent['th'] 
        movement_x = screen_x + line_length * np.cos(movedir)
        movement_y = screen_y - line_length * np.sin(movedir) 

        # This is the redline showing the player's moving direction, it's length represents the player's velocity magnitude
        pygame.draw.line(screen, (255, 0, 0), (int(screen_x), int(screen_y)), (int(movement_x), int(movement_y)), 5)

        # Showing ID
        font = pygame.font.SysFont(None, 40)
        text = font.render(str(agent['id']), True, (255, 255, 255))
        screen.blit(text, (int(screen_x) - 10, int(screen_y) - 10))
    
    ball_screen_x = SCREEN_WIDTH / 2 + ball_pos['x']
    ball_screen_y = SCREEN_HEIGHT / 2 - ball_pos['y']
    ball_rect = ball_image.get_rect(center=(ball_screen_x, ball_screen_y))
    screen.blit(ball_image, ball_rect.topleft)

def goto_position(world_index, agent_index, goal_position, desired_velocity, grid_world):
    # Get the agent's current position and facing angle
    x = grid_world.player_pos[0][agent_index][0]
    y = grid_world.player_pos[0][agent_index][1]
    v = grid_world.player_pos[0][agent_index][3]
    facing_angle = grid_world.player_pos[0][agent_index][5]

    # Compute the vector towards the goal
    dx = goal_position[0] - x
    dy = goal_position[1] - y

    # Compute the angle towards the goal
    desired_direction = np.arctan2(dy, dx)

    # Stop early if reaches goal already
    if v/2 < np.hypot(abs(dx),abs(dy)) < 5*v/6 :
        print("Yes, reached!")
        grid_world.actions[world_index, agent_index] = torch.tensor([0, 0, 0])
        return True

    # Normalize desired_direction to be between -pi and pi
    desired_direction = (desired_direction + np.pi) % (2 * np.pi) + np.pi

    # Compute the difference between current facing angle and desired direction
    angle_diff = desired_direction - facing_angle
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

    # Set angular velocity proportional to angle difference
    angular_velocity = angle_diff * 2.0  # Scaling factor

    grid_world.actions[world_index, agent_index] = torch.tensor([desired_velocity, desired_direction, angular_velocity])

    return False



# Right now, the code simply increments player position by 1 each loop
# the Player positions tensor is of shape, (num_worlds, num_players * 2)
# Where each pair of 2 elements (ex. index 2 and index 3) correspond to the x and y position of a player
# Its a hacked together solution, but hopefully the meeting tommorow can help with that. 
print(grid_world.player_pos)

frames = []

# Drill below
# Initialize agent states
agents_state = [{'state': 'waiting', 'timer': 0.0} for _ in range(10)]

# Initialize positions
spacing = 10.0  # No interval between players
line_start_x = -50.0  # Starting position at the free throw line (-10, 0)
line_y = 0.0

# Set initial positions for the line
for i in range(num_players):
    x = line_start_x + spacing * i
    y = line_y
    grid_world.player_pos[0][i][0] = x  # x position
    grid_world.player_pos[0][i][1] = y  # y position
    grid_world.player_pos[0][i][5] = 0.0  # facing angle

# Set first player to 'at_free_throw_line'
agents_state[0]['state'] = 'at_free_throw_line'
agents_state[0]['timer'] = 0.0

for idx in range(args.num_steps):

    if args.savevideo:
        frame_data = pygame.surfarray.array3d(screen)
        frames.append(frame_data.transpose((1, 0, 2)))  # Adjust to (width, height, channels)

    # set action tensor
    for j in range(num_worlds):
        # for i in range(num_players):
            # of shape (num_worlds, num_players, 3) where 3 is [acceleration, direction of accel, angular accel]
            
            # before: a, th, alpha
            # grid_world.actions[j, i] = torch.tensor([5.0, i * 0.5, (i-5) * 0.2])
            # grid_world.actions[j, i] = torch.tensor([0.0, i * 0.5, (i-5) * 0.2])

            # what we want: v, th(moving direction), omega (angular velocity of facing)
            # if i == 0:
            #     grid_world.actions[j,i] = goto_position(j,i,(0,0),10,grid_world)

        for agent_index in range(10):
            state = agents_state[agent_index]['state']
            timer = agents_state[agent_index]['timer']

            if state == 'waiting':
                # The agent is in line, remain stationary
                grid_world.actions[0, agent_index] = torch.tensor([0.0, 0.0, 0.0])

            elif state == 'at_free_throw_line':
                # The agent is at the free throw line, wait for 1 second
                agents_state[agent_index]['timer'] += dt
                grid_world.actions[0, agent_index] = torch.tensor([0.0, 0.0, 0.0])

                if agents_state[agent_index]['timer'] >= 1.0:
                    # After 1 second, change state to 'going_up'
                    agents_state[agent_index]['state'] = 'going_up'

            elif state == 'going_up':
                print("goingup")
                # The agent moves to (0, 20)
                goal_position = (0.0, 40.0)
                desired_velocity = 100.0  # Adjust as needed
                if goto_position(j,agent_index, goal_position, desired_velocity, grid_world):
                    agents_state[agent_index]['state'] = 'returning_to_line'
                    # time.sleep(0.3)
            elif state == 'returning_to_line':
                # The agent moves to the end of the line
                # The end of the line is at x = line_start_x + spacing * (num_players - 1)
                goal_x = line_start_x + spacing * (num_players - 1)
                goal_x += 10
                goal_position = (goal_x, 1.2585)
                desired_velocity = 100.0  # Adjust as needed
                if goto_position(j,agent_index, goal_position, desired_velocity, grid_world):
                    # Agent reached the end of the line
                    agents_state[agent_index]['state'] = 'waiting'
                    agents_state[agent_index]['completed'] = True
                    # Now, we need to scoot the other players to the left
                    agents_state[agent_index]['needs_scoot'] = True
                    # time.sleep(1)

            # After updating all agents, handle the scoot
            for agent_index in range(num_players):
                if agents_state[agent_index].get('needs_scoot', False):
                    # Perform scoot
                    # Define the scoot distance (e.g., the width of one player)
                    scoot_distance = 10.0  # Adjust this value as needed

                    # Move all agents (excluding the returning agent) one position to the left
                    for i in range(num_players):
                        # if i != agent_index:  # Exclude the returning agent who goes to the end
                        x = grid_world.player_pos[0][i][0]
                        grid_world.player_pos[0][i][0] = x - scoot_distance

                    # Set the next player to 'at_free_throw_line'
                    next_player_index = (agent_index + 1) % num_players  # Loop back if needed
                    agents_state[next_player_index]['state'] = 'at_free_throw_line'
                    agents_state[next_player_index]['timer'] = 0.0

                    agents_state[agent_index]['needs_scoot'] = False

            
    # Advance simulation across all worlds
    grid_world.step()
    print(grid_world.player_pos)
    print(grid_world.ball_pos)
    print(grid_world.who_holds)

    if args.logs and not args.visualize:
        with open(args.pos_logs_path, 'ab') as pos_logs:
            pos_logs.write(grid_world.player_pos.numpy().tobytes())
            pos_logs.write(grid_world.ball_pos.numpy().tobytes())
            pos_logs.write(grid_world.who_holds.numpy().tobytes())

    elif args.logs and args.visualize:
        
        agents = load_agents_from_tensor(grid_world.player_pos)  
        ballpos = load_ballpos_from_tensor(grid_world.ball_pos)
        whoholds = load_whoholds_from_tensor(grid_world.who_holds)

        screen.blit(court_img, (0, 0)) 
        draw_agents(screen, agents, ballpos) 
        pygame.display.flip() 
        time.sleep(0.1)

if args.savevideo and frames:
    # Set the filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"simulation_{timestamp}.mp4"
    
    # Set the frame rate (e.g., 10 FPS)
    clip = ImageSequenceClip(frames, fps=10)
    clip.write_videofile(video_filename, codec="libx264")


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