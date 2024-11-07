import numpy as np
from ._madrona_simple_example_cpp import SimpleGridworldSimulator, madrona

__all__ = ['GridWorld']

class GridWorld:
    def __init__(self,
                 initial_player_pos, # initial player positions
                 num_worlds,
                 start_cell,
                 end_cells,
                 rewards,
                 walls,
                 gpu_sim = False,
                 gpu_id = 0,
            ):
        self.size = np.array(walls.shape)
        self.start_cell = start_cell
        self.end_cells = end_cells
        self.rewards_input = rewards
        self.walls = walls
        self.court_size = np.array([94.0, 50.0]) # added court size, however it is not passed into madrona yet, TBD on use

        self.sim = SimpleGridworldSimulator(
                walls = np.array(walls).astype(np.bool_),
                rewards = np.array(rewards).astype(np.float32),
                end_cells = np.array(end_cells).astype(np.int32),
                init_player_pos = np.array(initial_player_pos).astype(np.float32), # give madrona initial positions
                start_x = start_cell[1],
                start_y = start_cell[0],
                max_episode_length = 0, # No max
                exec_mode = madrona.ExecMode.CUDA if gpu_sim else madrona.ExecMode.CPU,
                num_worlds = num_worlds, 
                num_players = len(initial_player_pos), #give madrona number of players with initial positions
                gpu_id = 0,
            )

        self.force_reset = self.sim.reset_tensor().to_torch()
        self.actions = self.sim.action_tensor().to_torch()
        self.observations = self.sim.observation_tensor().to_torch()
        self.rewards = self.sim.reward_tensor().to_torch()
        self.dones = self.sim.done_tensor().to_torch()
        self.player_pos = self.sim.player_tensor().to_torch() #new player position tensor
        self.ball_pos = self.sim.ball_tensor().to_torch()
        self.who_holds = self.sim.held_tensor().to_torch()

    def step(self):
        self.sim.step()
