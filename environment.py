import gym
import numpy as np
from util import gray_scale, resize, resize_gray
import PIL
from PIL import Image
from config import *
from atari_wrappers import *


class Env():
    def __init__(self, agent, env_id):
        env = gym.make(env_name)
        # env =  NoopResetEnv(env , noop_max=env_id)
        env = WarpFrame(env)
        env = FrameStack(env, k=k)
        self.env = env

        self.agent = agent
        self.env_id = env_id

    def preprocess(self, state):
        return state

    def run(self):
        np.random.seed(self.env_id)
        self.env.seed(self.env_id)

        # use to count episode
        count = 1
        state = self.env.reset()
        self.agent.reset()
        state = self.preprocess(state)
        one_episode_reward = 0
        # use to count step in one epoch
        step = 0
        while True:
            step += 1
            a = self.agent.choice_action(state)

            state_, r, done, info = self.env.step(a)

            one_episode_reward += r

            state_ = self.preprocess(state_)

            # This can limit max step
            # if step >= 60000:
            #    done = True

            self.agent.observe(state, a, r, state_, done)

            state = state_

            if done:
                print(str(self.env_id) + ":" + str(count) + "      :       " + str(one_episode_reward))
                count += 1
                one_episode_reward = 0
                state = self.env.reset()
                self.agent.reset()
                state = self.preprocess(state)
                step = 0
