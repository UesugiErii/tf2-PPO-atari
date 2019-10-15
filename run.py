from multiprocessing import Process
from communication import Communication
from config import *
from brain import ACBrain
from agent import Agent
from environment import Env
import subprocess
from util import get_seed


#   tensorboard --logdir logs/scalars

def main():
    communication = Communication(child_num=process_num)

    brain = ACBrain(talker=communication.master)

    envs_p = []

    seed = get_seed()
    for i in range(process_num):
        agent = Agent(talker=communication.children[i])
        env_temp = Env(agent, i, seed=seed+i)
        envs_p.append(Process(target=env_temp.run, args=()))

    for i in envs_p:
        i.start()

    tfb_p = subprocess.Popen(['tensorboard', '--logdir', "./logs/scalars"])

    brain.run()

    for p in envs_p:
        p.terminate()
    tfb_p.kill()


if __name__ == '__main__':
    main()
