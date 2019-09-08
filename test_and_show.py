import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
import time
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from atari_wrappers import *
from config import *


from model import ACModel

class ACAgent():
    def __init__(self, dir):
        super(ACAgent, self).__init__()
        self.Model = ACModel()
        self.Model.load_weights(dir)

    def choice_action(self, state):
        data = self.Model(np.array(state)[np.newaxis, :].astype(np.float32))
        prob_weights = data[0].numpy()

        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())

        return action


import gym

times = 100


class Env():
    def __init__(self, agent):
        env = gym.make(env_name)
        env = WarpFrame(env, width=IMG_W, height=IMG_H, grayscale=True)
        env = FrameStack(env, k=k)
        self.env = env

        self.agent = agent

    def preprocess(self, state):
        return state

    def run(self):
        count = 1
        np.random.seed(int(time.time()))
        self.env.seed(int(time.time()))
        state = self.env.reset()
        state = self.preprocess(state)
        one_episode_reward = 0
        step = 0
        l = []
        while 1:
            step += 1
            a = self.agent.choice_action(state)
            self.env.render()
            time.sleep(0.02)
            state_, r, done, info = self.env.step(a)

            one_episode_reward += r

            state_ = self.preprocess(state_)

            state = state_

            if done:
                print(":" + str(count) + "      :       " + str(one_episode_reward))
                count += 1
                l.append(one_episode_reward)
                if len(l) == times:
                    break
                one_episode_reward = 0
                state = self.env.reset()
                state = self.preprocess(state)
                step = 0
        print(sum(l) / times)


#
# for i in range(4, 5):
print('---------------------------------------------')
# print(i)
# index = 16000 * i
index = 64000
restore_weight_dir = "./logs/pong/{}".format(index)
Env(ACAgent(restore_weight_dir)).run()
print('---------------------------------------------')

# 48000         31.71
# 96000         178.86
# 144000        354.27
# 192000        390.44
