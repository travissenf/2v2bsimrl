import numpy as np
import json
import torch
from ._madrona_simple_example_cpp import SimpleGridworldSimulator, madrona

__all__ = ['GridWorld']
P_LOC_INDEX_TO_VAL = {0: "x", 1: "y", 2: "theta", 3: "velocity", 4:"angular v", 5: "facing angle"}
B_LOC_INDEX_TO_VAL = {0: "x", 1: "y", 2: "theta", 3: "velocity"}

class GridWorld:
    def __init__(self,
                 initial_player_pos, # initial player positions
                 num_worlds,
                 gpu_sim = False,
                 gpu_id = 0,
            ):
        self.court_size = np.array([94.0, 50.0]) # added court size, however it is not passed into madrona yet, TBD on use

        self.sim = SimpleGridworldSimulator(
                init_player_pos = np.array(initial_player_pos).astype(np.float32), # give madrona initial positions
                max_episode_length = 0, # No max
                exec_mode = madrona.ExecMode.CUDA if gpu_sim else madrona.ExecMode.CPU,
                num_worlds = num_worlds, 
                num_players = len(initial_player_pos), #give madrona number of players with initial positions
                gpu_id = 0,
            )

        self.actions = self.sim.action_tensor().to_torch()
        self.player_pos = self.sim.player_tensor().to_torch() #new player position tensor
        self.ball_pos = self.sim.ball_tensor().to_torch()
        self.who_holds = self.sim.held_tensor().to_torch()
        self.choices = self.sim.choice_tensor().to_torch()
        self.foul_call = self.sim.foul_call_tensor().to_torch()
        self.scoreboard = self.sim.scorecard_tensor().to_torch()
        self.resettens = self.sim.reset_tensor().to_torch()

    def step(self):
        self.sim.step()

    def reset(self, input_path):
        try:
            with open(input_path, 'r') as file:
                game_state = json.load(file)
            print(f"Game state has been loaded from {input_path}")
        except Exception as e:
            print(f"An error occurred while loading the file: {e}")
            return
        
        # Load players' data
        
        num_players = len(game_state["players"])
        
        player_pos_shape = self.player_pos[0].shape
        for i in range(num_players):
            self.actions[0][i][3] = 0.0
            self.actions[0][i][4] = 1.0
            self.choices[0][i][0] = 0
            self.foul_call[0][i][0] = 0
            for j in range(player_pos_shape[1]):
                key = P_LOC_INDEX_TO_VAL[j]
                self.player_pos[0][i][j] = game_state["players"][i][key]
                if (j == 2):
                    self.actions[0][i][1] = game_state["players"][i][key]
                elif (j == 3):
                    self.actions[0][i][0] = game_state["players"][i][key]
                elif (j == 4):
                    self.actions[0][i][2] = game_state["players"][i][key]
        
        for i in range(self.ball_pos[0].shape[0]):
            key = B_LOC_INDEX_TO_VAL[i]
            self.ball_pos[0][i] = game_state["ball"][key]
        
        self.who_holds[0][0] = game_state["ball"]["who holds"]
        self.who_holds[0][1] = game_state["ball"]["who shot"]
        self.who_holds[0][2] = -1
        if (game_state["ball"]["who holds"] > 0):
            self.who_holds[0][3] = 5
        else:
            self.who_holds[0][3] = 0
        self.scoreboard[0] = torch.tensor([0, 0, 1, 0])

        return {}
