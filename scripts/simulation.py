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
import tkinter as tk
from tkinter import messagebox
from policies import SimulationPolicies
import json

P_LOC_INDEX_TO_VAL = {0: "x", 1: "y", 2: "theta", 3: "velocity", 4:"angular v", 5: "facing angle"}
P_LOC_VAL_TO_INDEX = {"x": 0, "y": 1, "theta": 2, "velocity": 3, "angular v": 4, "facing angle": 5}

B_LOC_INDEX_TO_VAL = {0: "x", 1: "y", 2: "theta", 3: "velocity"}

SUPPORTED_POLICIES = {'run_in_line', 'run_and_defend'}

class Simulation(SimulationPolicies):
    def __init__(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--visualize', action='store_true', help="Enable visualization")
        arg_parser.add_argument('--logs', action='store_true', help="Enable logging")
        arg_parser.add_argument('--num_worlds', type=int, default=1)
        arg_parser.add_argument('--num_steps', type=int, default=1000)
        arg_parser.add_argument('--use_gpu', type=bool, default=False)
        arg_parser.add_argument('--pos_logs_path', type=str, default="pos_logs.bin")
        arg_parser.add_argument('--savevideo', action='store_true', help="Save each frame as an image for video creation")
        arg_parser.add_argument('--load_state', type=str, default=None, help="Load initial state json file from gamestates folder")
        arg_parser.add_argument('--policy', type=str, default='run_in_line', help="Pick which policy to run")
        self.args = arg_parser.parse_args()
        if (self.args.policy not in SUPPORTED_POLICIES):
            raise Exception("Invalid policy, does not exist")
        elif (self.args.policy == 'run_in_line'):
            self.initialize_policy = self.initialize_run_in_line
            self.run_policy = self.run_in_line_policy
        elif (self.args.policy == 'run_and_defend'):
            self.initialize_policy = self.run_around_and_defend_initialize
            self.run_policy = self.run_around_and_defend_policy
        # Constants
        self.PLAYER_CIRCLE_SIZE = 12
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 940, 500
        self.FEET_TO_PIXELS = 10.0  # 10 pixels per foot
        self.dt = 0.1
        self.num_players = 10
        self.players = []
        self.points = []
        self.frames = []
        self.num_worlds = self.args.num_worlds
        self.enable_gpu_sim = self.args.use_gpu
        self.grid_world = None
        background_img = cv2.imread("asset/warped_court.jpg")
        # Initialize time and score
        self.elapsed_time = 0.0  # Time in seconds
        self.score = [0, 0]      # Score initialized at 0:0
        self.show_details = False
        self.is_paused = False
        self.manipulation_mode = False
        self.selected_player = None
        self.current_viewed_world = 0

        self.background_surface = pygame.surfarray.make_surface(
            np.transpose(cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB), (1, 0, 2))
        )

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Basketball Court Simulation")

        # Load assets
        self._load_assets()
        self.view_angle = 0

    def _load_assets(self):
        # Load images and scale them
        self.pacman_yellow = pygame.image.load("asset/Pacman_yellow.png")
        self.pacman_blue = pygame.image.load("asset/Pacman_blue.png")
        self.ball_image = pygame.image.load("asset/ball.png")
        self.court_img = pygame.image.load("asset/court.png")

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
    def load_agents_from_tensor(self, tensor):
        worlds_agents = []
        
        for world_index, world_data in enumerate(tensor):
            # Load agents data
            agents = []
            
            for player_id in range(self.num_players): 
                agent_id = player_id
                # Extract x and y for the player
                x = float(world_data[player_id][0])
                y = float(world_data[player_id][1])
                th = float(world_data[player_id][2])
                v = float(world_data[player_id][3])
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
                    'id': agent_id,  # Increment ID by 1
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

            x = float(ball_data[0])
            y = float(ball_data[1])
            th = float(ball_data[2])
            v = float(ball_data[3])

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
    
    def display_time(self, screen):
        """
        Displays the elapsed time on the screen in format xx:xx.
        """
        # Convert elapsed time to minutes and seconds
        total_seconds = int(self.elapsed_time)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        time_str = f"{minutes:02d}:{seconds:02d}"  # Format as 'mm:ss'

        # Choose font and render the text
        font = pygame.font.SysFont(None, 40)
        time_text = font.render(time_str, True, (255, 255, 255))  # White color

        # Position the text on the screen (e.g., top center)
        time_rect = time_text.get_rect(center=(self.SCREEN_WIDTH // 2, 20))

        # Blit the text onto the screen
        screen.blit(time_text, time_rect)

    def display_score(self, screen):
        """
        Displays the current score on the screen in format x:x.
        """
        # Format the score
        score_str = f"{self.score[0]}:{self.score[1]}"

        # Choose font and render the text
        font = pygame.font.SysFont(None, 40)
        score_text = font.render(score_str, True, (255, 255, 255))  # White color

        # Position the text on the screen (e.g., top left corner)
        score_rect = score_text.get_rect(topleft=(20, 20))

        # Blit the text onto the screen
        screen.blit(score_text, score_rect)


    
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

    def draw_agents(self, screen, world, world_ball_position, view_angle):
        agents = world[0]['agents']
        ball_pos = world_ball_position[0]['ballpos']

        if view_angle == 0:
            # Original drawing method
            for agent in agents:
                # Using center circle as (0,0)
                screen_x = self.SCREEN_WIDTH / 2 + agent['x']* self.FEET_TO_PIXELS
                screen_y = self.SCREEN_HEIGHT / 2 - agent['y']* self.FEET_TO_PIXELS  # Y axis is opposite

                # Choose image based on agent ID
                if 0 <= agent['id'] < 5:
                    agent_image = self.pacman_yellow
                else:
                    agent_image = self.pacman_blue

                # Rotation angle is the player's facing angle
                rotated_image = pygame.transform.rotate(agent_image, np.degrees(agent['facing']))
                rotated_rect = rotated_image.get_rect(center=(screen_x, screen_y))
                screen.blit(rotated_image, rotated_rect.topleft)

                # Display parameters if show_details is True
                if self.show_details:
                    line1 = f"x:{agent['x']:.2f}, y:{agent['y']:.2f}, z:{agent.get('z', 0):.2f}"
                    line2 = f"th:{agent['th']:.2f}, v:{agent['v']:.2f}, facing:{agent['facing']:.2f}"

                    params_font = pygame.font.SysFont(None, 20)
                    line1_render = params_font.render(line1, True, (255, 255, 255))
                    line2_render = params_font.render(line2, True, (255, 255, 255))
                    
                    screen.blit(line1_render, (int(screen_x) - 50, int(screen_y) - 30))
                    screen.blit(line2_render, (int(screen_x) - 50, int(screen_y) - 15))


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
            ball_screen_x = self.SCREEN_WIDTH / 2 + ball_pos['x'] * self.FEET_TO_PIXELS
            ball_screen_y = self.SCREEN_HEIGHT / 2 - ball_pos['y'] * self.FEET_TO_PIXELS
            ball_rect = self.ball_image.get_rect(center=(ball_screen_x, ball_screen_y))
            screen.blit(self.ball_image, ball_rect.topleft)

        elif view_angle == 1:
            # Draw background once
            screen.blit(self.background_surface, (0, 0))

            for agent in agents:
                x = agent['x'] * self.FEET_TO_PIXELS + self.SCREEN_WIDTH/2
                y = -agent['y'] * self.FEET_TO_PIXELS + self.SCREEN_HEIGHT/3
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

                # Display parameters if show_details is True
                if self.show_details:
                    line1 = f"x:{agent['x']:.2f}, y:{agent['y']:.2f}, z:{agent.get('z', 0):.2f}"
                    line2 = f"th:{agent['th']:.2f}, v:{agent['v']:.2f}, facing:{agent['facing']:.2f}"

                    params_font = pygame.font.SysFont(None, 20)
                    line1_render = params_font.render(line1, True, (255, 255, 255))
                    line2_render = params_font.render(line2, True, (255, 255, 255))
                    
                    screen.blit(line1_render, (int(x_t) - 50, int(y_t) - 30))
                    screen.blit(line2_render, (int(x_t) - 50, int(y_t) - 15))


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

    def open_player_input_window(self, player_id):
        def on_ok():
            try:
                # Read and validate inputs
                values = [float(entry.get()) for entry in entries]
                for i in range(len(values)):
                    self.grid_world.player_pos[self.current_viewed_world][player_id][i] = values[i]
                # self.float_values = values  # Store the values
                popup.destroy()  # Close the popup
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid float values!")

        # Create the Tkinter popup window
        popup = tk.Tk()
        popup.title("Enter Float Values")

        tk.Label(popup, text="Enter 6 float values:").grid(row=0, column=0, columnspan=2, pady=10)

        entries = []
        for i in range(self.grid_world.player_pos[self.current_viewed_world][0].shape[0]):
            tk.Label(popup, text=f"{P_LOC_INDEX_TO_VAL[i]}: ").grid(row=i+1, column=0, padx=10, pady=5)
            entry = tk.Entry(popup)
            entry.insert(0, str(self.grid_world.player_pos[self.current_viewed_world][player_id][i].item()))
            entry.grid(row=i+1, column=1, padx=10, pady=5)
            entries.append(entry)

        tk.Button(popup, text="OK", command=on_ok).grid(row=7, column=0, columnspan=2, pady=10)

        popup.mainloop()

    def save_game_state(self, output):
        players_data = []
        for i in range(self.grid_world.player_pos[self.current_viewed_world].shape[0]):
            data = {"id": i}
            for j in range(self.grid_world.player_pos[self.current_viewed_world][0].shape[0]):
                data[f"{P_LOC_INDEX_TO_VAL[j]}"] = self.grid_world.player_pos[self.current_viewed_world][i][j].item()
            players_data.append(data)
        
        ball_data = {}
        for i in range(self.grid_world.ball_pos[self.current_viewed_world].shape[0]):
            ball_data[f"{B_LOC_INDEX_TO_VAL[i]}"] = self.grid_world.ball_pos[self.current_viewed_world][i].item()

        ball_data["who shot"] = self.grid_world.who_holds[self.current_viewed_world][1].item()
        ball_data["who holds"] = self.grid_world.who_holds[self.current_viewed_world][0].item()

        game_state = {
            "players": players_data,
            "ball": ball_data
        }
        
        try:
            with open(output, 'w') as file:
                json.dump(game_state, file, indent=4)
            print(f"Game state has been written to {output}")
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")

    def load_from_json(self, input_path):
        try:
            with open(input_path, 'r') as file:
                game_state = json.load(file)
            print(f"Game state has been loaded from {input_path}")
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")
            return
        
        # Load players' data
        num_players = len(game_state["players"])
        player_pos_shape = self.grid_world.player_pos[self.current_viewed_world].shape
        for i in range(num_players):
            for j in range(player_pos_shape[1]):
                key = P_LOC_INDEX_TO_VAL[j]
                self.grid_world.player_pos[self.current_viewed_world][i][j] = game_state["players"][i][key]

        # Load ball data
        for i in range(self.grid_world.ball_pos[self.current_viewed_world].shape[0]):
            key = B_LOC_INDEX_TO_VAL[i]
            self.grid_world.ball_pos[self.current_viewed_world][i] = game_state["ball"][key]
        
        self.grid_world.who_holds[self.current_viewed_world][0] = game_state["ball"]["who holds"]
        self.grid_world.who_holds[self.current_viewed_world][1] = game_state["ball"]["who shot"]

    def run(self):
        # Initialize agent states
        agents_state = self.initialize_policy()
        idx = 0
        if (self.args.load_state is not None):
            self.load_from_json("gamestates/" + self.args.load_state)
        
        while (idx < self.args.num_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.cleanup()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        self.manipulation_mode = not self.manipulation_mode
                        if self.manipulation_mode:
                            print("Entered manipulation mode.")
                        else:
                            print("Exited manipulation mode.")
                            self.selected_player = None

                    # Handle number keys '0' to '9' to select players
                    if self.manipulation_mode and event.key >= pygame.K_0 and event.key <= pygame.K_9:
                        self.selected_player = event.key - pygame.K_0  # Players 1-9 are indices 0-8
                        print(f"Selected player {self.selected_player}")

                    # Handle arrow keys to move the selected player
                    elif self.manipulation_mode and self.selected_player is not None and event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                        step_size = 3  # Adjust the step size as needed
                        if event.key == pygame.K_UP:
                            self.grid_world.player_pos[self.current_viewed_world][self.selected_player][1] += step_size
                        elif event.key == pygame.K_DOWN:
                            self.grid_world.player_pos[self.current_viewed_world][self.selected_player][1] -= step_size
                        elif event.key == pygame.K_LEFT:
                            self.grid_world.player_pos[self.current_viewed_world][self.selected_player][0] -= step_size
                        elif event.key == pygame.K_RIGHT:
                            self.grid_world.player_pos[self.current_viewed_world][self.selected_player][0] += step_size

                        if self.args.visualize:
                            # Clear the screen or blit background as needed
                            if self.view_angle == 0:
                                self.screen.blit(self.court_img, (0, 0))
                            elif self.view_angle == 1:
                                self.screen.blit(self.background_surface, (0, 0))

                            # Load agents and ball positions
                            agents = self.load_agents_from_tensor(self.grid_world.player_pos)
                            ballpos = self.load_ballpos_from_tensor(self.grid_world.ball_pos)

                            # Draw agents with the current view_angle
                            self.draw_agents(self.screen, agents, ballpos, self.view_angle)

                            # Display time and score
                            self.display_time(self.screen)
                            self.display_score(self.screen)

                            # Update the display immediately
                            pygame.display.flip()

                    elif event.key == pygame.K_v:
                        # Toggle view_angle between 0 and 1
                        self.view_angle = 0 if self.view_angle == 1 else 1
                        # Adjust screen size based on the new view angle
                        if self.view_angle == 0:
                            self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 940, 500
                        elif self.view_angle == 1:
                            self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 940, 688
                        
                        # Update the screen display size
                        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

                    elif event.key == pygame.K_d:
                        # Toggle show_details
                        self.show_details = not self.show_details
                    
                    elif event.key == pygame.K_s:
                        # Toggle show_details
                        self.grid_world.choices[self.current_viewed_world][self.grid_world.who_holds[self.current_viewed_world][0]][0] = 1

                    elif event.key == pygame.K_a:
                        # Toggle show_details
                        self.grid_world.choices[self.current_viewed_world][self.grid_world.who_holds[self.current_viewed_world][0]][0] = 2
                    
                    elif event.key == pygame.K_p:
                        self.is_paused = not self.is_paused

                    # Modify Travis's code, read player's status
                    elif event.key >= pygame.K_0 and event.key <= pygame.K_9 and not self.manipulation_mode:
                        self.open_player_input_window(event.key - pygame.K_0)

                        if self.args.visualize:
                            # Clear the screen or blit background as needed
                            if self.view_angle == 0:
                                self.screen.blit(self.court_img, (0, 0))
                            elif self.view_angle == 1:
                                self.screen.blit(self.background_surface, (0, 0))

                            # Load agents and ball positions
                            agents = self.load_agents_from_tensor(self.grid_world.player_pos)
                            ballpos = self.load_ballpos_from_tensor(self.grid_world.ball_pos)

                            # Draw agents with the current view_angle
                            self.draw_agents(self.screen, agents, ballpos, self.view_angle)

                            # Display time and score
                            self.display_time(self.screen)
                            self.display_score(self.screen)

                            # Update the display immediately
                            pygame.display.flip()
                    
                    elif event.key == pygame.K_s:
                        output_file = "gamestates/" + input("Enter the filename to save the JSON (e.g., 'game_state'): ").strip() + ".json"

                        # Default filename if the user doesn't enter anything
                        if not output_file:
                            output_file = "gamestates/game_state.json"

                        self.save_game_state(output_file)

            
            # if self.is_paused:
            #     continue

            if not self.is_paused:
                agents_state = self.run_policy(agents_state)

                ## CODE TO SHOOT / PASS
                # if ((idx == 20) and (self.grid_world.who_holds[self.current_viewed_world][0] != -1)):
                #     self.grid_world.choices[self.current_viewed_world][self.grid_world.who_holds[self.current_viewed_world][0]][0] = 1
                
                # if ((idx == 20) and (self.grid_world.who_holds[self.current_viewed_world][0] != -1)):
                #     self.grid_world.choices[self.current_viewed_world][self.grid_world.who_holds[self.current_viewed_world][0]][0] = 2
            
                
                t = time.time()
                self.grid_world.step()
                for i in range(self.num_players):
                    self.grid_world.choices[self.current_viewed_world][i][0] = 0
                self.elapsed_time += 0.1
                idx += 1
            else:
                # When paused, ensure actions are zeroed out
                self.grid_world.actions.zero_()
                continue

            agents_state = self.run_policy(agents_state)

            if self.args.savevideo:
                frame_data = pygame.surfarray.array3d(self.screen)
                frame_data = frame_data.transpose((1, 0, 2))
                standard_size = (940, 688)
                frame_data_resized = cv2.resize(frame_data, standard_size, interpolation=cv2.INTER_LINEAR)
                self.frames.append(frame_data_resized)
            

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

                # Display time and score
                self.display_time(self.screen)
                self.display_score(self.screen)

                pygame.display.flip()
                if (time.time() - t < 0.1):
                    time.sleep(0.1 - (time.time() - t))
                

        if self.args.savevideo:
            frame_data = pygame.surfarray.array3d(self.screen)
            frame_data = frame_data.transpose((1, 0, 2))
            standard_size = (940, 688)
            frame_data_resized = cv2.resize(frame_data, standard_size, interpolation=cv2.INTER_LINEAR)
            self.frames.append(frame_data_resized)


        if self.args.savevideo and self.frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"videos/simulation_{timestamp}.mp4"
            clip = ImageSequenceClip(self.frames, fps=10)
            clip.write_videofile(video_filename, codec="libx264")


    def cleanup(self):
        pygame.quit()
        sys.exit()
