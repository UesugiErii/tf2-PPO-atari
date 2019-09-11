import numpy as np
import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
import tensorflow.keras.optimizers as optim
from config import *
from model import ACModel


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr):
        super(CustomSchedule, self).__init__()
        self.last_lr = init_lr
        self.max_learning_times = max_learning_times

    def __call__(self, step):
        # every time call , step automatic += 1
        temp_lr = lr * ((self.max_learning_times - step) / self.max_learning_times)
        self.last_lr = temp_lr
        return temp_lr

    def get_config(self):
        return self.last_lr


class ACBrain():
    def __init__(self, talker):
        super(ACBrain, self).__init__()
        self.model = ACModel()
        self.model.build((None, IMG_H, IMG_W, k))
        self.talker = talker
        self.i = 1
        self.optimizer = optim.Adam(learning_rate=CustomSchedule(lr))
        self.states_list = self.talker.states_list
        self.memory = []
        self.one_episode_reward_index = 0

    def forward_calc(self, state):
        state = state.astype(np.float32)
        a_prob, v = self.model.predict(state)
        return [a_prob, v]  # [(*,a_num) , (*,1)]  np.array

    # states
    # for master            # Now dont use it
    #  0          1
    # work      train
    # for child
    #  0          1
    # work     finished
    def run(self):
        print("brain" + "      ", os.getpid())
        while 1:
            while 1:

                # why use [0] * n
                # This can pre-allocate memory , save time at copy data from a old small list to a new big list

                n = process_num - sum(self.states_list[1:])  # how much data will I receive next
                temp_data = [0] * n  # use to recode data that send by child
                temp_id = [0] * n  # use to recode which child send data to brain
                count = 0  # if one is prepare to learn , then need pop temp_*
                for i in range(n):
                    child_id, data = self.talker.recv()
                    flag, origin_data = data  # flag  0 means learning data 1 means predict data
                    if flag:
                        temp_data[count] = origin_data
                        temp_id[count] = child_id
                        count += 1
                    else:
                        self.talker.states_list[child_id] = 1  # means this agent wait for learning
                        episode_reward = origin_data.pop()  # fetch one_episode_reward

                        for one_episode_reward in episode_reward:  # use tensorflow recode reward in one episode
                            self.model.record(name='one_episode_reward', data=one_episode_reward,
                                              step=self.one_episode_reward_index)
                            self.one_episode_reward_index += 1

                        self.memory.append(origin_data)  # store learning data

                        del temp_data[-1]
                        del temp_id[-1]
                if all(self.states_list[1:]):  # all agents wait for learning
                    break
                data = np.stack(temp_data, axis=0)
                res = self.forward_calc(data)
                for i, child_id in enumerate(temp_id):
                    self.talker.send(
                        [res[0][i], res[1][i]],
                        child_id
                    )

            self.learn()
            for child_id in range(1, process_num + 1):  # tell agents that can start act with env
                self.states_list[child_id] = 0
                self.talker.send("ok", child_id)

    def learn(self):
        total_obs = []
        total_as = []
        total_old_ap = []
        total_adv = np.array([], dtype=np.float32)
        total_real_v = np.array([], dtype=np.float32)

        # Data preprocessing before learning
        # realv means a state's target v

        for data in self.memory:
            ep_obs, ep_as, realv, adv, ep_old_ap = data
            total_as.extend(ep_as)
            total_old_ap.extend(ep_old_ap)
            total_real_v = np.concatenate([total_real_v, realv])
            total_adv = np.concatenate([total_adv, adv])
            total_obs.extend(ep_obs)

        total_obs = np.stack(total_obs, axis=0).astype(np.float32)
        total_old_ap = np.stack(total_old_ap, axis=0)
        total_as = tf.one_hot(total_as, depth=a_num).numpy()

        for _ in range(epochs):
            sample_index = np.random.choice(total_as.shape[0], size=learning_batch)
            grads, loss = self.model.total_grad(total_obs[sample_index],
                                                total_as[sample_index],
                                                total_adv[sample_index],
                                                total_real_v[sample_index],
                                                total_old_ap[sample_index])
            grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.i += 1
        print("-------------------------")
        print(self.i)
        # if self.i == max_learning_times-5:
        #     import sys
        #     self.talker.close_all()       # send kill signal to child(agent)
        #     sys.exit(0)

        self.memory = []
