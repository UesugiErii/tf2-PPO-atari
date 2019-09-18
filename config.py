process_num = 16                             # number of actors
IMG_H = 84                                   # height of state after processing
IMG_W = 84                                   # width of state after processing
batch_size = 64                              # Rollout length , horizon , how much step in one agent in one learn time
recode_span = 50                             # tensorflow record span
save_span = 16000                            # after (save_span//epoch) update parameter , save parameter , for example if save_span=16000 , epoch=3 , after about 16000/3 will save model parameter
beta = 0.01                                  # Entropy coeff
clip_epsilon = 0.1                           # clip ε , I dont use annealed
lr = 0.00025                                 # learning rate
max_learning_times = 200000*3                # max learning time
gamma = 0.99                                 # discount reward
learning_batch = process_num*batch_size//4   # learn batch
epochs = 3                                   # learning epochs time
VFcoeff = 1                                  # same as PPO paper

env_name = 'PongDeterministic-v4'            # env name
# env_name = 'BreakoutDeterministic-v4'            # env name
import gym
env = gym.make(env_name)
a_num = env.action_space.n                   # env.action_space
del env

use_RNN = True
hidden_unit_num = 128
if use_RNN:
    k = 1                                    # can be other number
else:
    k = 4                                    # frame stack number
