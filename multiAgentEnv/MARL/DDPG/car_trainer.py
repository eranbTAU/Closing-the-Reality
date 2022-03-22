import time
import numpy as np
from utils import save, load, modify_action, dict_to_np
import matplotlib.pyplot as plt
import os

from maenv.utils import average_total_reward

class Trainer(object):
    def __init__(self, params):
        self.gen_params = params['gen_params']
        self.env_params = params['env_params']
        self.rew_scaler = self.env_params['reward_scaler']
        self.agent_params = params['agent_params']


        self.logger = Logger(params, self.rew_scaler)

        self.episodes = self.env_params['episodes']
        self.max_steps = self.env_params['max_steps']
        self.render = self.env_params['render']


    def run(self, env, agent):
        if self.gen_params['is_train']:
            print('starting training mode...')
            self.run_train_loop(env, agent)
        else:
            print('starting eval mode...')
            self.policy_eval(env, agent)

    def run_train_loop(self, env, agent):
        if self.agent_params['load_policy'] is not None:
            print('loading saved plicy models...')
            agent.load_models()

        for episode in range(1, self.episodes + 1):
            agent.noise.reset()
            obss = env.reset()
            for t in range(self.max_steps):
                if self.render:
                    env.render()
                actions = agent.choose_action(dict_to_np(obss, np.float32))
                obss_, rewards, dones, infos = env.step(actions)
                for a in env.possible_agents:
                    agent.store_experience(obss[a], actions[a], rewards[a]/self.env_params['reward_scaler'], obss_[a], dones[a])
                q_value = agent.learn()
                self.logger(actions, rewards, dones, infos, q_value)
                obss = obss_
                if any(dones.values()):
                    break
                if 'out' in infos.values():
                    break

            self.logger.reset_episode()
            if self.logger.avg_score > self.logger.best_score:
                agent.save_models()
            env.close()

        self.logger.finish()


    def policy_eval(self, env, agent):
        raise NotImplementedError


class Logger():
    def __init__(self, params, rew_baseline):
        self.params = params
        # env
        self.num_agents = params['env_params']['num_agents']
        self.max_steps = params['env_params']['max_steps']
        self.episodes = params['env_params']['episodes']
        self.is_training = params['gen_params']['is_train']
        self.rew_scaler = self.params['env_params']['reward_scaler']

        # counters
        self.episode = 1
        self.steps = 0
        self.total_steps = 0
        self.start_time = time.time()
        self.elapsed_time = 0

        # metrics
        if not self.is_training:
            self.report_every = 1
        else:
            self.report_every = params['gen_params']['report_interval']

        self.episode_return = 0
        self.ep_returns = []
        self.best_score = -1000
        self.avg_score = -1000

        # plots
        self.dir = params['gen_params']['save_dir']
        self.dir_name = os.path.split(self.dir)[1]
        self.train_steps = []

        # logs
        self.episode_actions = []
        self.action_log = []
        self.episode_rewards = []
        self.reward_log = []
        self.episode_infos = []
        self.info_log = []
        self.episode_q = []
        self.q_log = []

        self.log_filename = os.path.join(self.dir, ('train' if self.is_training else 'eval') + '_log')

    def __call__(self, actions, rewards, dones, infos, q):
        self.steps += 1
        self.total_steps += 1
        self.episode_return += np.average(list(rewards.values()))/self.rew_scaler
        self.log(actions, rewards, dones, infos, q) # each episode


    def log(self, actions, rewards, dones, infos, q):
        # every step:
        rewards = list(rewards.values())
        self.episode_rewards.append(rewards)
        self.episode_q.append(q)


    def reset_episode(self):
        self.ep_returns.append(self.episode_return)
        self.reward_log.append(self.episode_rewards)
        self.q_log.append(self.episode_q)

        # every end of episode
        if self.episode % 20 == 0:  # save every 10 episodes and reset
            if self.episode == 20:
                log = {}
                # log['actions'] = self.action_log
                log['rewards'] = self.reward_log
                # log['infos'] = self.info_log
                log['scores'] = self.ep_returns
                log['q_values'] = self.q_log
            else:
                log = load(self.log_filename+'.pkl')
                log['rewards'] += self.reward_log
                log['scores'] = self.ep_returns
                log['q_values'] += self.q_log
            save(log, self.log_filename)
                           
            self.reward_log = []
            self.q_log = []

        self.report()

        self.episode_return = 0
        self.episode += 1
        self.steps = 0
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_infos = []
        self.episode_q = []

    def report(self): # after each episode
        self.avg_score = np.mean(self.ep_returns[-self.report_every:])
        if self.avg_score > self.best_score:
            self.best_score = self.avg_score-1e-5

        self.train_steps.append(self.total_steps)
        self.reward_log.append(self.episode_rewards)

        now = time.time()
        if self.episode % self.report_every == 0: # after report_every episodes

            last_elapsed = round((now - self.start_time) / 60, 1)

            plot_learn_curve(self.ep_returns,
                             self.train_steps,
                             10,
                             os.path.join(self.dir, ('train' if self.is_training else 'eval')  + '_score_vs_steps_' + self.dir_name),
                             True)

            self.elapsed_time += last_elapsed
            self.start_time = now
            print(
                f'episode: {self.episode} | last {self.report_every} episodes average score: {np.round(self.avg_score, 2)} | {last_elapsed} [minutes]')

    def finish(self):
        name = os.path.split(self.dir)[0]
        plot_learn_curve(self.ep_returns,
                         self.train_steps,
                         10,
                         os.path.join(self.dir, ('train' if self.is_training else 'eval') + '_score_vs_steps_' + self.dir_name),
                         True)

        print(f'saved in {name}')
        print('%d [hours] in total' % (round(self.elapsed_time / 60, 2)))

def plot_learn_curve(scores, x, window_size, figure_file, add_std):
    running_average = np.zeros(len(scores))
    std = np.zeros(len(scores))
    for i,_ in enumerate(running_average):
        running_average[i] = np.mean(scores[max(0,i-window_size):(i+1)])
        if add_std:
            std[i] = np.std(scores[max(0,i-window_size):(i+1)])

    fig, ax = plt.subplots(figsize=(10, 5))
    # ax.set_ylim(bottom=-1, top=0)
    ax.plot(x, running_average)
    if add_std:
        ax.fill_between(x, running_average - std, running_average + std,
                         color='gray', alpha=0.2)
    plt.title(f'running average over {window_size} last time-steps')
    plt.savefig(figure_file+'.png')
