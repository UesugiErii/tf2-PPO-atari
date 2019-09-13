import numpy as np
from config import *
import sys


class Agent():
    def __init__(self, talker, seed):
        super(Agent, self).__init__()
        self.talker = talker

        self.episode_reward = []
        self.send_as = np.zeros((batch_size,), dtype=np.float32)
        self.send_rs = np.zeros((batch_size,), dtype=np.float32)
        self.send_is_done = np.zeros((batch_size,), dtype=np.float32)
        self.index = 0
        self.one_episode_reward = 0
        np.random.seed(seed)

    def choice_action(self, state):
        self.talker.send(state)
        prob_weights = self.talker.recv()  # a_prob

        action = np.random.choice(range(a_num), p=prob_weights)

        # print prob to check the code for errors
        # if after a while , one action prob always is near 1 , there must have error in code
        if np.random.random() < 0.001:
            print(prob_weights, action)

        return action

    def observe(self, state, a, R, state_, done):
        self.send_as[self.index] = a
        self.send_is_done[self.index] = done
        self.one_episode_reward += R
        self.send_rs[self.index] = R
        self.index += 1

        if done:
            self.episode_reward.append(self.one_episode_reward)
            self.one_episode_reward = 0

        if self.index % batch_size == 0:
            # print('reach horizon')
            self.choice_action(state_)
            # print('send all data')
            self.send_all_data()
            self.index = 0

    def send_all_data(self):
        self.talker.send(
            [
                self.send_as,
                self.send_rs,
                self.send_is_done,
                self.episode_reward
            ]
        )
        self.episode_reward = []

        if self.talker.recv() == 'ok':  # sync message , tell agent you can start act with env
            pass
        else:
            raise RuntimeError("DONT RECV ok")

