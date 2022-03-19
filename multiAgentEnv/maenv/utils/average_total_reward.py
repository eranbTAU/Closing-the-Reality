import random

import numpy as np


def average_total_reward(env, max_episodes=100, max_steps=10000000000):
    '''
    Edited for parallel_env(!)

    Runs an env object with random actions until either max_episodes or
    max_steps is reached. Calculates the average total reward over the
    episodes.

    Reward is summed across all agents, making it unsuited for use in zero-sum
    games.
    '''
    total_reward = 0
    total_steps = 0
    for episode in range(max_episodes):
        env.reset()
        for t in range(max_steps):
            if total_steps >= max_steps:
                break
            actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
            obs, reward, done, infos = env.step(actions)
            total_reward += np.average(list(reward.values()))
            total_steps += 1
            env.render()
            if any(done.values()):
                num_episodes = episode + 1
                print('done', episode)
                break
            if 'out' in infos.values():
                print('out', episode)
                break
        env.close()

    print(f'Average agent [reward/episode]: {round(total_reward / num_episodes, 2)} '
          f'| in {num_episodes} episodes and total {total_steps} steps')
    return total_reward / num_episodes


# ----------------------------------------------
#         env.reset()
#         for agent in env.agent_iter():
#             obs, reward, done, _ = env.last(observe=False)
#             total_reward += reward
#             total_steps += 1
#             if done:
#                 action = None
#             elif isinstance(obs, dict) and 'action_mask' in obs:
#                 action = random.choice(np.flatnonzero(obs['action_mask']))
#             else:
#                 action = env.action_space(agent).sample()
#             env.step(action)
#
#         num_episodes = episode + 1
#     print("Average total reward", total_reward / num_episodes)
#
#     return total_reward / num_episodes
