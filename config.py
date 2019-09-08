process_num = 16                             # number of actors
IMG_H = 84                                   # height of state after processing
IMG_W = 84                                   # width of state after processing
batch_size = 64                              # horizon , how much step in one agent in one learn time
recode_span = 50                             # tensorflow record span
beta = 0.01                                  # Entropy coeff
lr = 0.00025                                 # learning rate
max_learning_times = 200000*3                # max learning time
gamma = 0.99                                 # discount reward
k = 4                                        # frame stack number
learning_batch = process_num*batch_size//4   # learn batch
epochs = 3                                   # learning epochs time
VFcoeff = 1                                  # same as PPO paper
env_name = 'PongDeterministic-v4'            # env name

# import gym
# env = gym.make(env_name)
# a_num = env.action_space.n                 # env.action_space
# del env

a_num = 4                                    # sometime need set