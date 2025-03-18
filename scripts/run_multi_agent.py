import sys
import numpy as np
import torch
from madrona_simple_example import GridWorld
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms import Algorithm
from ray import tune
from ray.tune.registry import register_env
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from multi_agent_train import BasketballMultiAgentEnv  # Import from your new file
import matplotlib.pyplot as plt
from ray.tune.logger import TBXLoggerCallback

if __name__ == "__main__":
    ray.init()

    config = PPOConfig()

    config = config.debugging(log_level="INFO", log_sys_usage=True)

    print("ENVIRONMENT")
    config = config.environment(BasketballMultiAgentEnv, env_config={"reset_path": "gamestates/2v2init.json"})
    print("MULTI AGENT")
    config = config.multi_agent(
            # Define two policies.
            policies={"offense", "defense"},
            policy_mapping_fn=lambda agent_id, episode, **kw: agent_id,
        ).training(entropy_coeff=0.002)
    print("TORCH FRAMEWORK")
    config = config.framework("torch")

    config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)

    offense_rewards = [] 

    print('Building Trainer')
    trainer = config.build_algo()
    trainer.remove_policy("offense")
    trainer.remove_policy("defense")
    trainer.add_policy("offense", policy=Policy.from_checkpoint(r"C:\Users\travi\repos\madrona_simple_example\scripts\checkpoints\evenbettermodels\iter_950\policies\offense"))
    trainer.add_policy("defense", policy=Policy.from_checkpoint(r"C:\Users\travi\repos\madrona_simple_example\scripts\checkpoints\evenbettermodels\iter_950\policies\defense"))
    # trainer.load_checkpoint(r'C:\Users\travi\repos\madrona_simple_example\scripts\checkpoints\evenbettermodels\iter_650')
    print('Starting Training')
    for i in range(651, 10001):
        results = trainer.train()
        print(f"Iteration: {i}")

        offense_reward = results['env_runners']['policy_reward_mean']['offense']
        offense_rewards.append(offense_reward)  

        print(f"Environment steps sampled throughput (steps/sec): {results['num_env_steps_sampled_throughput_per_sec']}")
        print(f"Environment steps trained throughput (steps/sec): {results['num_env_steps_trained_throughput_per_sec']}")
        print(f"Time for this iteration (s): {results['time_this_iter_s']}")
        print(f"Number of steps trained this iteration: {results['num_steps_trained_this_iter']}")

        offense_reward = results['env_runners']['policy_reward_mean']['offense']  
        defense_reward = results['env_runners']['policy_reward_mean']['defense']  
        print(f"Offense policy reward mean: {offense_reward}")
        print(f"Defense policy reward mean: {defense_reward}")
        if i % 50 == 0:
            checkpoint_dir = f'checkpoints/evenbettermodels/iter_{i}'
            trainer.save(checkpoint_dir)
            print(f"Checkpoint saved at {checkpoint_dir}")

        if (i != 0 and i % 50 == 0):
            plt.plot(offense_rewards)
            plt.title("Offense Reward Over Time")
            plt.xlabel("Step (Iteration)")
            plt.ylabel("Offense Reward Mean")
            plt.savefig(f"offense_reward_plot_even_better_iter_{i}.png")

    ray.shutdown()