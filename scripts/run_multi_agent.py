import sys
import numpy as np
import torch
from madrona_simple_example import GridWorld
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.tune.registry import register_env
import gymnasium
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from multi_agent_train import BasketballMultiAgentEnv  # Import from your new file
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ray.init()

    config = PPOConfig()
    print("ENVIRONMENT")
    config = config.environment(BasketballMultiAgentEnv, env_config={"reset_path": "gamestates/2v2init.json"})
    print("MULTI AGENT")
    config = config.multi_agent(
            # Define two policies.
            policies={"offense", "defense"},
            policy_mapping_fn=lambda agent_id, episode, **kw: agent_id,
        )
    print("TORCH FRAMEWORK")
    config = config.framework("torch")

    config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)

    offense_rewards = [] 

    print('Building Trainer')
    trainer = config.build_algo()
    print('Starting Training')
    for i in range(50):
        results = trainer.train()
        print(f"Iteration: {i}")

        offense_reward = results['env_runners']['policy_reward_mean']['offense']
        offense_rewards.append(offense_reward)  

        print(f"Environment steps sampled throughput (steps/sec): {results['num_env_steps_sampled_throughput_per_sec']}")
        print(f"Environment steps trained throughput (steps/sec): {results['num_env_steps_trained_throughput_per_sec']}")
        print(f"Time for this iteration (s): {results['time_this_iter_s']}")
        print(f"Number of steps trained this iteration: {results['num_steps_trained_this_iter']}")

        # Accessing the policy rewards for offense and defense from env_runners
        offense_reward = results['env_runners']['policy_reward_mean']['offense']  # Assuming offense is at index 0
        defense_reward = results['env_runners']['policy_reward_mean']['defense']  # Assuming defense is at index 1

        print(f"Offense policy reward mean: {offense_reward}")
        print(f"Defense policy reward mean: {defense_reward}")
        if i % 10 == 0:
            checkpoint_dir = f'checkpoints/iter_{i}'
            trainer.save(checkpoint_dir)
            print(f"Checkpoint saved at {checkpoint_dir}")

    plt.plot(offense_rewards)
    plt.title("Offense Reward Over Time")
    plt.xlabel("Step (Iteration)")
    plt.ylabel("Offense Reward Mean")
    plt.savefig("offense_reward_plot_first.png")

    ray.shutdown()