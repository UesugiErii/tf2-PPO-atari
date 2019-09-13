from multiprocessing import Queue, Value, Array, Pipe, Lock
from multiprocessing import Process
from config import *


class Master():
    def __init__(self, send_queues, recv_queue, states_list):
        self.send_queues = send_queues  # [pipe , pipe , pipe .........]
        self.recv_queue = recv_queue  # pipe
        self.states_list = states_list

    def send(self, data, child_id: int):
        self.send_queues[child_id].send(data)
        return 1

    def recv(self):
        return self.recv_queue.recv()

    def set_state(self, state: int):
        self.states_list[0] = state

    def close_all(self):
        for pipe in self.send_queues:
            pipe.close()


class Child():
    def __init__(self, lock, send_queue, recv_queue, states_list, child_id):
        self.lock = lock
        self.send_queue = send_queue  # pipe
        self.recv_queue = recv_queue  # pipe
        self.states_list = states_list
        self.child_id = child_id

    # send game state or game memory to master to calc
    def send(self, data):
        self.lock.acquire()
        try:
            self.send_queue.send(
                (self.child_id, data)
            )
        finally:
            self.lock.release()

    # child must can recv data after send data to master
    def recv(self):
        try:
            data = self.recv_queue.recv()
        except:
            print("master is end")
            raise Exception
        return data

    def set_state(self, state: int):
        # WARNING : child only can change his state
        self.states_list[self.child_id] = state

    def get_state(self):
        return self.states_list[self.child_id]


class Communication():
    """
    this class use to let master process and child processes communicate
    only one master
    """

    def __init__(self, child_num: int):
        # in master perspective

        send_s_pipe, recv_s_pipe = Pipe()
        send_res_pipes = []
        recv_res_pipes = []
        lock = Lock()
        for i in range(child_num):
            s, r = Pipe()
            send_res_pipes.append(s)
            recv_res_pipes.append(r)

        # states
        # for child
        #  0          1
        # work     finished
        # if all children is finished, then master can start learning

        # children states list
        states_list = Array('i', [0 for _ in range(child_num)])

        self.master = Master(send_queues=send_res_pipes,
                             recv_queue=recv_s_pipe,
                             states_list=states_list
                             )

        self.children = [0]*child_num
        for i in range(child_num):
            self.children[i] = Child(
                lock=lock,
                send_queue=send_s_pipe,
                recv_queue=recv_res_pipes[i],
                states_list=states_list,  #
                child_id=i  # child_id  from 0 to child_num-1
            )
