import numpy as np
import time
from config import *
import sys


class Agent():
    def __init__(self, talker, seed):
        super(Agent, self).__init__()
        self.talker = talker
        self.states_list = self.talker.states_list
        self.reset()

        # In send span , may have more than one episode
        # send_*  use to store data that given to brain to learn
        self.episode_reward = []
        self.send_obs = []
        self.send_as = []
        self.send_advs = []
        self.send_realvs = []
        self.send_old_ap = []
        self.index = 0
        np.random.seed(seed)

    def choice_action(self, state):
        self.talker.send([1, state])  # when choice action,also calc v  1 means data use to predict
        data = self.talker.recv()

        if data == -1:
            sys.exit(0)
        # accroding data to choise action
        # data contain actions prob and v
        self.ep_vs.append(data[1])

        prob_weights = data[0]

        action = np.random.choice(range(a_num),
                                  p=prob_weights)

        self.ep_old_ap.append(prob_weights)

        # print prob to check the code for errors
        # if after a while , one action prob always is near 1 , there must have error in code
        # if np.random.random() < 0.001:
        #     print(prob_weights, action)

        return action

    def observe(self, state, a, R, state_, done):
        """
            two situations:

            (1): done , but data dont have reach batch_size(horizon)

                start a new episode

                move self.ep_* to self.send_* (also need calc adv,realv)

            (2): reach batch_size(horizon)

                calc realv and adv , then send data

            done and reach batch_size(horizon) , this no need to consider
        """

        self.ep_obs.append(state)
        self.ep_as.append(a)
        self.ep_rs.append(R)
        self.one_episode_reward += R
        self.index += 1
        if done:
            self.episode_reward.append(self.one_episode_reward)
            self.send_obs.extend(self.ep_obs)
            self.send_as.extend(self.ep_as)
            self.ep_vs.append(np.zeros((1, 1)))
            realv, adv = self.calc_realv_adv(self.ep_rs, self.ep_vs)
            self.send_advs.append(adv)
            self.send_realvs.append(realv)
            self.send_old_ap.extend(self.ep_old_ap)

            self.ep_obs = []
            self.ep_as = []
            self.ep_rs = []
            self.ep_vs = []
            self.ep_old_ap = []

        if self.index % batch_size == 0:
            if len(self.send_as) != batch_size:
                self.choice_action(state_)
                self.ep_old_ap.pop()

                self.send_obs.extend(self.ep_obs)
                self.send_as.extend(self.ep_as)

                realv, adv = self.calc_realv_adv(self.ep_rs, self.ep_vs)
                self.send_advs.append(adv)
                self.send_realvs.append(realv)
                self.send_old_ap.extend(self.ep_old_ap)

            assert len(self.send_as) == batch_size

            self.send_all_data()
            self.index = 0

            self.ep_obs = []
            self.ep_as = []
            self.ep_rs = []
            self.ep_vs = []
            self.ep_old_ap = []

    def send_all_data(self):
        self.talker.send([
            0,  # 0 means this data is used to learn
            [
                self.send_obs,  # list array
                self.send_as,  # list int
                np.concatenate(self.send_realvs),  # list array -> array(batch_size,)
                np.concatenate(self.send_advs),  # list array -> array(batch_size,)
                np.stack(self.send_old_ap, axis=0),  # list array -> array(batch_size,action_num)
                self.episode_reward  # list int
            ]
        ])
        self.episode_reward = []
        self.send_obs = []
        self.send_as = []
        self.send_advs = []
        self.send_realvs = []
        self.send_old_ap = []
        if self.talker.recv() == 'ok':  # sync message , tell agent you can start act with env
            pass
        else:
            raise RuntimeError("DONT RECV ok")


    def calc_realv_adv(self, ep_rs, ep_vs):
        """
        :param ep_rs: list n   float
        :param ep_vs: list n+1 np.array (1,)
        :return: np.array (n,) float32 , np.array (n,) float32

        r1 r2 r3
        v1 v2 v3 end_v
        """
        length = len(ep_rs)
        realv = np.zeros((length + 1,), dtype=np.float32)
        realv[-1] = ep_vs[-1]
        adv = np.zeros((length,), dtype=np.float32)

        for t in range(length - 1, -1, -1):
            realv[t] = realv[t + 1] * gamma + ep_rs[t]  # realv means q(a,s) = r + 0.99*v(t+1)
            adv[t] = realv[t] - ep_vs[t]

        return realv[:-1], adv  # end_v dont need

    def reset(self):
        # ep_* use to store data in one episode
        self.ep_obs = []
        self.ep_as = []
        self.ep_rs = []
        self.ep_vs = []
        self.ep_old_ap = []

        self.one_episode_reward = 0
