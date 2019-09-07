import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
import time
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from atari_wrappers import *
from config import *


class ACModel(Model):
    def __init__(self, name, dir):
        self.n = name
        super(ACModel, self).__init__()
        self.c1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                         activation='relu')
        self.c2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu')
        self.c3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(512, activation="relu")
        self.d2 = Dense(1)  # C
        self.d3 = Dense(4, activation='softmax')  # A
        self.a_index = 0
        self.c_index = 0
        self.save_index = 0
        self.call(np.random.random((1, IMG_H, IMG_W, k)).astype(np.float32))
        self.load_weights(dir)

    @tf.function
    def call(self, inputs):
        x = inputs / 255.0
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.flatten(x)
        x = self.d1(x)
        a = self.d3(x)
        c = self.d2(x)
        return a, c


class ACAgent():
    def __init__(self, dir):
        super(ACAgent, self).__init__()
        self.Model = ACModel("model", dir=dir)

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
index = 0
restore_weight_dir = "./logs/weight/{}".format(index)
Env(ACAgent(restore_weight_dir)).run()
print('---------------------------------------------')

# 48000         31.71
# 96000         178.86
# 144000        354.27
# 192000        390.44
