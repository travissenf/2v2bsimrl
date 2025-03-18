import numpy as np
import torch
from madrona_simple_example import GridWorld
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from gymnasium import spaces



P_LOC_INDEX_TO_VAL = {0: "x", 1: "y", 2: "theta", 3: "velocity", 4:"angular v", 5: "facing angle"}
P_LOC_VAL_TO_INDEX = {"x": 0, "y": 1, "theta": 2, "velocity": 3, "angular v": 4, "facing angle": 5}
B_LOC_INDEX_TO_VAL = {0: "x", 1: "y", 2: "theta", 3: "velocity"}

class BasketballMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        
        self.reset_path = config.get("reset_path")
        self.seed = 42
        points = []
        for i in range(4):
            points.append([(i - 5) * 5, (i - 5) * 5, 0, 0.0, 0.0, -np.pi])

        print(f"Initializing New Basketball World in worker {ray.get_runtime_context().get_worker_id()}")
        grid_world = GridWorld(points, 1, False, 0)  
        grid_world.reset(self.reset_path)
    
        self.grid_world = grid_world
        self.agents = self.possible_agents = ["offense", "defense"]
        
        self.seed = 42
        self._agent_ids = {"offense", "defense"}

        # Define action spaces
        offense_action_space = spaces.Dict({
            "player1": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "player2": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "decision": spaces.Discrete(3)
        })
        
        defense_action_space = spaces.Dict({
            "player1": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "player2": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        })
        
        # Define observation spaces
        offense_obs_space = spaces.Dict({
            "player_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(24, ), dtype=np.float32),
            "ball_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            "who_holds": spaces.Discrete(5),
            "who_shot": spaces.Discrete(5),
            "who_passed": spaces.Discrete(5),
            "ball_state": spaces.Discrete(6),
            "scoreboard": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.int32)
        })
        
        defense_obs_space = spaces.Dict({
            "player_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(24, ), dtype=np.float32),
            "ball_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            "who_holds": spaces.Discrete(5),
            "who_shot": spaces.Discrete(5),
            "who_passed": spaces.Discrete(5),
            "ball_state": spaces.Discrete(6),
            "scoreboard": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.int32)
        })
        
        # Public attributes for action and observation spaces
        self.action_spaces = {
            "offense": offense_action_space,
            "defense": defense_action_space
        }
        
        self.observation_spaces = {
            "offense": offense_obs_space,
            "defense": defense_obs_space
        }

    def reset(self, *, seed=None, options=None):
        """Reset the environment according to Gymnasium API"""
        if seed is not None:
            np.random.seed(seed)
        self.grid_world.reset(self.reset_path)
        obs = self._get_obs()
        return obs, {}

    def step(self, action_dict):
        """Step the environment according to Gymnasium API"""
        # Process actions
        offense_action = action_dict["offense"]
        defense_action = action_dict["defense"]

        off_act1 = np.array(offense_action["player1"]).copy()
        off_act2 = np.array(offense_action["player2"]).copy()

        def_act1 = np.array(defense_action["player1"]).copy()
        def_act2 = np.array(defense_action["player2"]).copy()

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
        self.grid_world.actions[0][2] = torch.tensor(defense_action["player1"])
        self.grid_world.actions[0][3] = torch.tensor(def_act2)
        self.grid_world.choices[0] = torch.tensor(
            [offense_action["decision"]] * 2 + [0] * 2
        ).view(4, 1)
        
        # Step the simulation
        self.grid_world.step()

        # Get observations and rewards
        obs = self._get_obs()
        rewards = self._compute_rewards()
        
        # Check for termination conditions
        is_terminated = (
            self.grid_world.scoreboard[0][0].item() != 0 or
            abs(self.grid_world.ball_pos[0][0].item()) > 47.0 or
            abs(self.grid_world.ball_pos[0][1].item()) > 25.0 or
            self.grid_world.who_holds[0][0].item() > 1 or
            self.grid_world.foul_call[0].numpy().sum() != 0 or 
            self.grid_world.scoreboard[0][3].item() >= 20 / 0.05 or
            self.grid_world.ball_pos[0][0] > 0.0
        )
        
        # Create required dictionaries
        terminateds = {"__all__": is_terminated}
        truncateds = {"__all__": is_terminated}
        
        return obs, rewards, terminateds, truncateds, {}
    
    def _get_obs(self):
        """Helper method to get observations for all agents"""
        shared_obs = {
            "player_pos": np.float32(self.grid_world.player_pos[0].numpy()).flatten(),
            "ball_pos": np.float32(self.grid_world.ball_pos[0].numpy()),
            "who_holds": int(np.add(self.grid_world.who_holds[0][0].numpy(), [1])[0]),
            "who_shot": int(np.add(self.grid_world.who_holds[0][1].numpy(), [1])[0]),
            "who_passed": int(np.add(self.grid_world.who_holds[0][2].numpy(), [1])[0]),
            "ball_state": int(self.grid_world.who_holds[0].numpy()[3]),
            "scoreboard": self.grid_world.scoreboard[0].numpy()
        }
        shared_obs2 = {
            "player_pos": np.float32(self.grid_world.player_pos[0].numpy()).flatten(),
            "ball_pos": np.float32(self.grid_world.ball_pos[0].numpy()),
            "who_holds": int(np.add(self.grid_world.who_holds[0][0].numpy(), [1])[0]),
            "who_shot": int(np.add(self.grid_world.who_holds[0][1].numpy(), [1])[0]),
            "who_passed": int(np.add(self.grid_world.who_holds[0][2].numpy(), [1])[0]),
            "ball_state": int(self.grid_world.who_holds[0].numpy()[3]),
            "scoreboard": self.grid_world.scoreboard[0].numpy()
        }
        return {
            "offense": shared_obs,
            "defense": shared_obs2
        }
    
    def _compute_rewards(self):
        """Helper method to compute rewards for all agents"""
        reward_offense = 0.0
        reward_defense = 0.0
        
        if self.grid_world.scoreboard[0][0].item() != 0:
            reward_offense += self.grid_world.scoreboard[0][0].item() * 5
            reward_defense -= self.grid_world.scoreboard[0][0].item() * 5
        
            
        if abs(self.grid_world.ball_pos[0][0]) > 47.0 or abs(self.grid_world.ball_pos[0][1]) > 25 or self.grid_world.ball_pos[0][0] > 0.0 or self.grid_world.ball_pos[0][1] > 0.0:
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
        
        if self.grid_world.scoreboard[0][3].item() >= 20 / 0.05:
            reward_defense += 3
            reward_offense -= 3
            
        return {
            "offense": reward_offense,
            "defense": reward_defense
        }