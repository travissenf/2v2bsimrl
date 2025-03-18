from hybrid_sac import HybridSAC
from sac import SAC
from madrona_simple_example import GridWorld
import numpy as np
import torch

MAX_STEPS = 20 * 20*10
MAX_EPISODES = 1000
EVAL_INTERVAL = 10
REWARD_SCALE = 15


if __name__ == "__main__":
    points = []
    for i in range(4):
        points.append([(i - 5) * 5, (i - 5) * 5, 0, 0.0, 0.0, -np.pi])

    grid_world = GridWorld(points, 1, False, 0)  
    
    state_dim = 4*6 + 4 + 5 + 5 + 5 + 6 + 4
    cont_action_dim = 2 * 5
    disc_action_dim = 3
    
    defense_agent = SAC(state_dim=state_dim, 
                        action_dim=cont_action_dim,
                        hidden_dim=256,
                        auto_entropy_tuning=True)
    
    offense_agent = HybridSAC(state_dim=state_dim, 
                              continuous_dim=cont_action_dim, 
                              discrete_dim=disc_action_dim)
    
    defense_episode_rewards = []
    offense_episode_rewards = []
    
    for episode in range(MAX_EPISODES):
        grid_world.reset("gamestates/2v2init.json")
        shared_obs = {
            "player_pos": np.float32(grid_world.player_pos[0].numpy()).flatten(),
            "ball_pos": np.float32(grid_world.ball_pos[0].numpy()),
            "who_holds": int(np.add(grid_world.who_holds[0][0].numpy(), [1])[0]),
            "who_shot": int(np.add(grid_world.who_holds[0][1].numpy(), [1])[0]),
            "who_passed": int(np.add(grid_world.who_holds[0][2].numpy(), [1])[0]),
            "ball_state": int(grid_world.who_holds[0].numpy()[3]),
            "scoreboard": grid_world.scoreboard[0].numpy()
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

        state = torch.tensor(np.concatenate(final_obs_list))
        
        for step in range(MAX_STEPS):
            defense_action = defense_agent.select_action(state)
            offense_cont_action, offense_disc_action = offense_agent.select_action(state)

            array1, array2 = torch.tensor_split(offense_cont_action[0], 2)
            off_act1, off_act2 = array1.numpy(), array2.numpy()

            array1, array2 = torch.tensor_split(defense_action, 2)
            def_act1, def_act2 = array1.numpy(), array2.numpy()

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

            grid_world.actions[0][0] = torch.tensor(off_act1)
            grid_world.actions[0][1] = torch.tensor(off_act2)
            grid_world.actions[0][2] = torch.tensor(def_act1)
            grid_world.actions[0][3] = torch.tensor(def_act2)
            grid_world.choices[0] = torch.tensor(
                [offense_disc_action] * 2 + [0] * 2
            ).view(4, 1)
            #UPDATE SIM W/ ACTIONS
            
            # Take action in environment
            grid_world.step()

            # GET STATE, STORE IN NEXT_STATE

            shared_obs = {
                "player_pos": np.float32(grid_world.player_pos[0].numpy()).flatten(),
                "ball_pos": np.float32(grid_world.ball_pos[0].numpy()),
                "who_holds": int(np.add(grid_world.who_holds[0][0].numpy(), [1])[0]),
                "who_shot": int(np.add(grid_world.who_holds[0][1].numpy(), [1])[0]),
                "who_passed": int(np.add(grid_world.who_holds[0][2].numpy(), [1])[0]),
                "ball_state": int(grid_world.who_holds[0].numpy()[3]),
                "scoreboard": grid_world.scoreboard[0].numpy()
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

            next_state = torch.tensor(np.concatenate(final_obs_list))


            done = (
                grid_world.scoreboard[0][0].item() != 0 or
                abs(grid_world.ball_pos[0][0]) > 47.0 or 
                abs(grid_world.ball_pos[0][1]) > 25 or 
                grid_world.ball_pos[0][0] > 0.0 or 
                grid_world.ball_pos[0][1] > 0.0 or
                grid_world.who_holds[0][0].item() > 1 or
                grid_world.foul_call[0].numpy().sum() != 0 or 
                grid_world.scoreboard[0][3].item() >= 20 / 0.05 or
                grid_world.ball_pos[0][0] > 0.0
            )
            
            reward_offense = 0.0
            reward_defense = 0.0
            if done:
                if grid_world.scoreboard[0][0].item() != 0:
                    reward_offense += grid_world.scoreboard[0][0].item() * 5
                    reward_defense -= grid_world.scoreboard[0][0].item() * 5
                
                if abs(grid_world.ball_pos[0][0]) > 47.0 or abs(grid_world.ball_pos[0][1]) > 25 or grid_world.ball_pos[0][0] > 0.0 or grid_world.ball_pos[0][1] > 0.0:
                    reward_offense -= 3
                    reward_defense += 3
                
                if grid_world.who_holds[0][0] > 1:
                    reward_defense += 3
                    reward_offense -= 3
                
                if grid_world.foul_call[0][0].item() != 0 or grid_world.foul_call[0][1].item() != 0:
                    reward_defense += 5
                    reward_offense -= 5

                if grid_world.foul_call[0][2].item() != 0 or grid_world.foul_call[0][3].item() != 0:
                    reward_defense -= 4
                    reward_offense += 4
                
                if grid_world.scoreboard[0][3].item() >= 20 / 0.05:
                    reward_defense += 3
                    reward_offense -= 3
                
                offense_episode_rewards.append(reward_offense + REWARD_SCALE)
                defense_episode_rewards.append(reward_defense + REWARD_SCALE)

            # Store transition in replay buffer
            defense_agent.replay_buffer.push(state, defense_action, reward_defense, next_state, done)
            offense_agent.replay_buffer.push(state, offense_cont_action, offense_disc_action, reward_offense, next_state, done)
            
            # Update parameters
            defense_agent.update_parameters()
            offense_agent.update_parameters()
            
            state = next_state
            if done:              
                break
        
        print(f"Episode {episode}: offense reward = {offense_episode_rewards[-1]}, defense reward = {defense_episode_rewards[-1]}")
    
    
    offense_agent.save("offense_agent_model_test.pth")
    defense_agent.save("defense_agent_model_test.pth")

    