# simulation.py
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

class Simulation:
    def __init__(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--visualize', action='store_true', help="Enable visualization")
        arg_parser.add_argument('--logs', action='store_true', help="Enable logging")
        arg_parser.add_argument('--num_worlds', type=int, default=1)
        arg_parser.add_argument('--num_steps', type=int, default=300)
        arg_parser.add_argument('--use_gpu', type=bool, default=False)
        arg_parser.add_argument('--pos_logs_path', type=str, default="pos_logs.bin")
        arg_parser.add_argument('--savevideo', action='store_true', help="Save each frame as an image for video creation")
        self.args = arg_parser.parse_args()

        # Constants
        self.PLAYER_CIRCLE_SIZE = 15
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 940, 500
        self.dt = 0.1
        self.num_players = 10
        self.players = []
        self.points = []
        self.frames = []
        self.num_worlds = self.args.num_worlds
        self.enable_gpu_sim = self.args.use_gpu
        self.grid_world = None
        background_img = cv2.imread("warped_court.jpg")

        self.background_surface = pygame.surfarray.make_surface(
            np.transpose(cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB), (1, 0, 2))
        )

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Basketball Court Simulation")

        # Load assets
        self._load_assets()
        self.view_angle = 1

    def _load_assets(self):
        # Load images and scale them
        self.pacman_yellow = pygame.image.load("Pacman_yellow.png")
        self.pacman_blue = pygame.image.load("Pacman_blue.png")
        self.ball_image = pygame.image.load("ball.png")
        self.court_img = pygame.image.load("court.png")

        self.pacman_yellow = pygame.transform.scale(self.pacman_yellow, (self.PLAYER_CIRCLE_SIZE * 4, self.PLAYER_CIRCLE_SIZE * 4))
        self.pacman_blue = pygame.transform.scale(self.pacman_blue, (self.PLAYER_CIRCLE_SIZE * 4, self.PLAYER_CIRCLE_SIZE * 4))
        self.ball_image = pygame.transform.scale(self.ball_image, (self.PLAYER_CIRCLE_SIZE * 2, self.PLAYER_CIRCLE_SIZE * 2))
        self.court_img = pygame.transform.scale(self.court_img, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.pacman_yellow_v = pygame.transform.scale(self.pacman_yellow, (self.PLAYER_CIRCLE_SIZE * 2, self.PLAYER_CIRCLE_SIZE * 2))
        self.pacman_blue_v = pygame.transform.scale(self.pacman_blue, (self.PLAYER_CIRCLE_SIZE * 2, self.PLAYER_CIRCLE_SIZE * 2))
        self.ball_image_v = pygame.transform.scale(self.ball_image, (self.PLAYER_CIRCLE_SIZE * 1, self.PLAYER_CIRCLE_SIZE * 1))

    def initialize_simulation(self):

        # Initial player positions
        for i in range(self.num_players):
            self.points.append([(i - 5) * 5, (i - 5) * 5, 0, 0.0, 0.0, -np.pi])

        # Create simulator object
        self.grid_world = GridWorld(self.points, self.num_worlds, self.enable_gpu_sim, 0)

        
    # Asumming tensor is (num_worlds, 2 * num_players)
    # Modified Jermaine's Code
    def load_agents_from_tensor(self,tensor):
        worlds_agents = []
        
        for world_index, world_data in enumerate(tensor):
            # Load agents data
            agents = []
            
            for player_id in range(self.num_players): 
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

    def load_ballpos_from_tensor(self,tensor):
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

    def load_whoholds_from_tensor(self,tensor):
        worlds_whoholds = []
        
        for world_index, whoholds_data in enumerate(tensor):

            whoholds_idx = int(whoholds_data[0])


            whoholds = {
                'whoholds': whoholds_idx
            }

            worlds_whoholds.append({'world_index': world_index, 'whoheld': whoholds})
        
        return worlds_whoholds
    
    def draw_pacman(self, screen, x, y, z, agent_image):
        """
        Draws the pacman image and its shadow on the screen.
        :param screen: Pygame display window
        :param x, y, z: The character's position in 3D space
        :param pacman_img: The image of the pacman to draw
        """
        # Perspective transformation matrix M
        M = np.array([[4.44444444e-01, -4.17333333e-01, 2.60833333e+02],
                    [1.51340306e-17, -1.11111111e-01, 3.47222222e+02],
                    [-0.00000000e+00, -8.88888889e-04, 1.00000000e+00]])

        y_shadow = y
        y += 125
        y_shadow += 125

        point = np.array([[x, y]], dtype='float32')
        point = np.array([point])

        point_shadow = np.array([[x, y_shadow]], dtype='float32')
        point_shadow = np.array([point_shadow])

        transformed_point = cv2.perspectiveTransform(point, M)
        transformed_point_shadow = cv2.perspectiveTransform(point_shadow, M)

        x_t, y_t = transformed_point[0][0]
        y_t -= z

        x_t_shadow, y_t_shadow = transformed_point_shadow[0][0]

        # Compute scaling ratio for Pacman image
        scale_ratio = 1  # Adjust as needed
        pacman_img = agent_image
        pacman_original_width, pacman_original_height = pacman_img.get_size()
        new_width = int(pacman_original_width * scale_ratio)
        new_height = int(pacman_original_height * scale_ratio)

        # Ensure new size is at least 5 pixels to prevent crash
        new_width = max(new_width, 5)
        new_height = max(new_height, 5)

        # Scale Pacman image
        scaled_pacman = pygame.transform.smoothscale(pacman_img, (new_width, new_height))

        # Adjust position to align image center with (x_t, y_t)
        pacman_pos = (int(x_t - new_width / 2), int(y_t - new_height / 2))

        # Draw shadow
        z1 = 0
        z2 = 500
        shadow_alpha = max(int(255 * (1 - (z - z1) / (z2 - z1))), 50)
        shadow_width = new_width
        shadow_height = max(int(new_height / 10), 3)

        shadow_surface = pygame.Surface((shadow_width, shadow_height), pygame.SRCALPHA)
        shadow_surface.set_alpha(shadow_alpha)
        pygame.draw.ellipse(shadow_surface, (0, 0, 0), (0, 0, shadow_width, shadow_height))
        shadow_pos = (int(x_t_shadow - shadow_width / 2), int(y_t_shadow + new_height / 2))
        screen.blit(shadow_surface, shadow_pos)

        # Draw Pacman image
        screen.blit(scaled_pacman, pacman_pos)

        # Return the transformed position for further use (e.g., to draw the agent ID)
        return x_t, y_t


    # def draw_agents(self, screen, world, world_ball_position, view_angle=0):
    #     agents = world[0]['agents']
    #     ball_pos = world_ball_position[0]['ballpos']

    #     for agent in agents:
    #         # Using center circle as (0,0)
    #         screen_x = self.SCREEN_WIDTH / 2 + agent['x']
    #         screen_y = self.SCREEN_HEIGHT / 2 - agent['y']  # Y axis is opposite

    #         # Choose image based on agent ID
    #         if 0 <= agent['id'] <= 5:
    #             agent_image = self.pacman_yellow
    #         else:
    #             agent_image = self.pacman_blue

    #         # Rotation angle is the player's facing angle
    #         rotated_image = pygame.transform.rotate(agent_image, np.degrees(agent['facing']))
    #         rotated_rect = rotated_image.get_rect(center=(screen_x, screen_y))
    #         screen.blit(rotated_image, rotated_rect.topleft)

    #         line_length = agent['v']

    #         movedir = agent['th'] 
    #         movement_x = screen_x + line_length * np.cos(movedir)
    #         movement_y = screen_y - line_length * np.sin(movedir) 

    #         # This is the redline showing the player's moving direction, it's length represents the player's velocity magnitude
    #         pygame.draw.line(screen, (255, 0, 0), (int(screen_x), int(screen_y)), (int(movement_x), int(movement_y)), 5)

    #         # Showing ID
    #         font = pygame.font.SysFont(None, 40)
    #         text = font.render(str(agent['id']), True, (255, 255, 255))
    #         screen.blit(text, (int(screen_x) - 10, int(screen_y) - 10))
        
    #     ball_screen_x = self.SCREEN_WIDTH / 2 + ball_pos['x']
    #     ball_screen_y = self.SCREEN_HEIGHT / 2 - ball_pos['y']
    #     ball_rect = self.ball_image.get_rect(center=(ball_screen_x, ball_screen_y))
    #     screen.blit(self.ball_image, ball_rect.topleft)

    def draw_agents(self, screen, world, world_ball_position, view_angle):
        agents = world[0]['agents']
        ball_pos = world_ball_position[0]['ballpos']

        if view_angle == 0:
            # Original drawing method
            for agent in agents:
                # Using center circle as (0,0)
                screen_x = self.SCREEN_WIDTH / 2 + agent['x']
                screen_y = self.SCREEN_HEIGHT / 2 - agent['y']  # Y axis is opposite

                # Choose image based on agent ID
                if 0 <= agent['id'] <= 5:
                    agent_image = self.pacman_yellow
                else:
                    agent_image = self.pacman_blue

                # Rotation angle is the player's facing angle
                rotated_image = pygame.transform.rotate(agent_image, np.degrees(agent['facing']))
                rotated_rect = rotated_image.get_rect(center=(screen_x, screen_y))
                screen.blit(rotated_image, rotated_rect.topleft)

                line_length = agent['v']

                movedir = agent['th']
                movement_x = screen_x + line_length * np.cos(movedir)
                movement_y = screen_y - line_length * np.sin(movedir)

                # This is the redline showing the player's moving direction, its length represents the player's velocity magnitude
                pygame.draw.line(screen, (255, 0, 0), (int(screen_x), int(screen_y)), (int(movement_x), int(movement_y)), 5)

                # Showing ID
                font = pygame.font.SysFont(None, 40)
                text = font.render(str(agent['id']), True, (255, 255, 255))
                screen.blit(text, (int(screen_x) - 10, int(screen_y) - 10))

            # Drawing the ball
            ball_screen_x = self.SCREEN_WIDTH / 2 + ball_pos['x']
            ball_screen_y = self.SCREEN_HEIGHT / 2 - ball_pos['y']
            ball_rect = self.ball_image.get_rect(center=(ball_screen_x, ball_screen_y))
            screen.blit(self.ball_image, ball_rect.topleft)

        elif view_angle == 1:
            # Draw background once
            screen.blit(self.background_surface, (0, 0))

            for agent in agents:
                x = agent['x'] + self.SCREEN_WIDTH/2
                y = -agent['y'] + self.SCREEN_HEIGHT/3
                z = agent.get('z', 0)  # Assuming z is available in agent data, default to 0 if not

                # Choose pacman image based on agent ID
                if 0 <= agent['id'] <= 5:
                    pacman_img = self.pacman_yellow_v
                else:
                    pacman_img = self.pacman_blue_v

                # Draw pacman and get transformed position
                x_t, y_t = self.draw_pacman(screen, x, y, z, pacman_img)

                # Show agent ID near the transformed point
                font = pygame.font.SysFont(None, 20)
                text = font.render(str(agent['id']), True, (255, 255, 255))
                screen.blit(text, (int(x_t) - 10, int(y_t) - 10))

            # Drawing the ball
            # Assuming the ball image is stored in ball_image variable
            x = ball_pos['x'] + self.SCREEN_WIDTH/2
            y = -ball_pos['y'] + self.SCREEN_HEIGHT/4
            z = ball_pos.get('z', 0)  # Assuming z coordinate is available, default to 0

            # Transform the ball position
            y_shadow = y
            y += 125
            y_shadow += 125

            point = np.array([[x, y]], dtype='float32')
            point = np.array([point])

            point_shadow = np.array([[x, y_shadow]], dtype='float32')
            point_shadow = np.array([point_shadow])

            M = np.array([[4.44444444e-01, -4.17333333e-01, 2.60833333e+02],
                        [1.51340306e-17, -1.11111111e-01, 3.47222222e+02],
                        [-0.00000000e+00, -8.88888889e-04, 1.00000000e+00]])

            transformed_point = cv2.perspectiveTransform(point, M)
            transformed_point_shadow = cv2.perspectiveTransform(point_shadow, M)

            x_t, y_t = transformed_point[0][0]
            y_t -= z

            x_t_shadow, y_t_shadow = transformed_point_shadow[0][0]

            # Scaling the ball image
            scale_ratio = 0.5  # Adjust as needed
            ball_original_width, ball_original_height = self.ball_image.get_size()
            new_width = int(ball_original_width * scale_ratio)
            new_height = int(ball_original_height * scale_ratio)

            new_width = max(new_width, 5)
            new_height = max(new_height, 5)

            scaled_ball = pygame.transform.smoothscale(self.ball_image_v, (new_width, new_height))

            ball_pos_screen = (int(x_t - new_width / 2), int(y_t - new_height / 2))

            # Draw ball shadow
            z1 = 0
            z2 = 500
            shadow_alpha = max(int(255 * (1 - (z - z1) / (z2 - z1))), 50)
            shadow_width = new_width
            shadow_height = max(int(new_height / 10), 3)

            shadow_surface = pygame.Surface((shadow_width, shadow_height), pygame.SRCALPHA)
            shadow_surface.set_alpha(shadow_alpha)
            pygame.draw.ellipse(shadow_surface, (0, 0, 0), (0, 0, shadow_width, shadow_height))
            shadow_pos = (int(x_t_shadow - shadow_width / 2), int(y_t_shadow + new_height / 2))
            screen.blit(shadow_surface, shadow_pos)

            # Draw ball image
            screen.blit(scaled_ball, ball_pos_screen)


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
        if v/2 < np.hypot(abs(dx),abs(dy)) < 5*v/6 :
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

    def run(self):
        # Initialize agent states
        agents_state = [{'state': 'waiting', 'timer': 0.0} for _ in range(self.num_players)]

        # Initialize positions
        spacing = 10.0  # No interval between players
        line_start_x = -50.0  # Starting position at the free throw line (-10, 0)
        line_y = 0.0

        # Set initial positions for the line
        for i in range(self.num_players):
            x = line_start_x + spacing * i
            y = line_y
            self.grid_world.player_pos[0][i][0] = x  # x position
            self.grid_world.player_pos[0][i][1] = y  # y position
            self.grid_world.player_pos[0][i][5] = 0.0  # facing angle

        # Set first player to 'at_free_throw_line'
        agents_state[0]['state'] = 'at_free_throw_line'
        agents_state[0]['timer'] = 0.0

        for idx in range(self.args.num_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.cleanup()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v:
                        # Toggle view_angle between 0 and 1
                        self.view_angle = 0 if self.view_angle == 1 else 1

            if self.args.savevideo:
                frame_data = pygame.surfarray.array3d(self.screen)
                self.frames.append(frame_data.transpose((1, 0, 2)))  # Adjust to (width, height, channels)

            # Set action tensor
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
                        goal_position = (0.0, 40.0)
                        desired_velocity = 100.0
                        if self.goto_position(j, agent_index, goal_position, desired_velocity):
                            agents_state[agent_index]['state'] = 'returning_to_line'

                    elif state == 'returning_to_line':
                        goal_x = line_start_x + spacing * (self.num_players - 1) + 10
                        goal_position = (goal_x, 1.2585)
                        desired_velocity = 100.0
                        if self.goto_position(j, agent_index, goal_position, desired_velocity):
                            agents_state[agent_index]['state'] = 'waiting'
                            agents_state[agent_index]['completed'] = True
                            agents_state[agent_index]['needs_scoot'] = True

                    for agent_index in range(self.num_players):
                        if agents_state[agent_index].get('needs_scoot', False):
                            scoot_distance = 10.0
                            for i in range(self.num_players):
                                x = self.grid_world.player_pos[0][i][0]
                                self.grid_world.player_pos[0][i][0] = x - scoot_distance

                            next_player_index = (agent_index + 1) % self.num_players
                            agents_state[next_player_index]['state'] = 'at_free_throw_line'
                            agents_state[next_player_index]['timer'] = 0.0
                            agents_state[agent_index]['needs_scoot'] = False

            # Advance simulation across all worlds
            self.grid_world.step()

            if self.args.logs:
                # Write log data to the file
                with open(self.args.pos_logs_path, 'ab') as pos_logs:
                    pos_logs.write(self.grid_world.player_pos.numpy().tobytes())
                    pos_logs.write(self.grid_world.ball_pos.numpy().tobytes())
                    pos_logs.write(self.grid_world.who_holds.numpy().tobytes())

                # Print log data to the terminal
                print("Logging Simulation State:")
                print(f"Player Positions:\n{self.grid_world.player_pos.numpy()}")
                print(f"Ball Position:\n{self.grid_world.ball_pos.numpy()}")
                print(f"Who Holds:\n{self.grid_world.who_holds.numpy()}")

            if self.args.visualize:
                # Visualization logic
                agents = self.load_agents_from_tensor(self.grid_world.player_pos)
                ballpos = self.load_ballpos_from_tensor(self.grid_world.ball_pos)
                whoholds = self.load_whoholds_from_tensor(self.grid_world.who_holds)

                # Clear the screen or blit background as needed
                if self.view_angle == 0:
                    self.screen.blit(self.court_img, (0, 0))
                elif self.view_angle == 1:
                    self.screen.blit(self.background_surface, (0, 0))

                # Draw agents with the current view_angle
                self.draw_agents(self.screen, agents, ballpos, self.view_angle)
                pygame.display.flip()
                time.sleep(0.1)



        if self.args.savevideo and self.frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"simulation_{timestamp}.mp4"
            clip = ImageSequenceClip(self.frames, fps=10)
            clip.write_videofile(video_filename, codec="libx264")


    def cleanup(self):
        pygame.quit()
        sys.exit()
