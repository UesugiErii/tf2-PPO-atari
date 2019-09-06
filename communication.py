from multiprocessing import Queue, Value, Array, Pipe, Lock
from multiprocessing import Process
from config import *


class Master():
    def __init__(self, send_queues, recv_queue, states_list):
        self.send_queues = send_queues  # [pipe , pipe , pipe .........]
        self.recv_queue = recv_queue  # pipe
        self.states_list = states_list

    def send(self, data, child_id: int):
        self.send_queues[child_id - 1].send(data)
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
            data = -1
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
        # for master
        #  0          1
        # work      train
        # for child
        #  0          1
        # work     finished
        # if all children is finished, then master can start learning
        # when master state from 1 to 0 , the children can restart play

        # 0 is master , 1~child_num is children
        states_list = []
        for i in range(1 + child_num):
            states_list.append(Value('d', 0))

        states_list = Array('i', [0 for _ in range(1 + child_num)])

        self.master = Master(send_queues=send_res_pipes,
                             recv_queue=recv_s_pipe,
                             states_list=states_list
                             )

        self.children = [0 for _ in range(child_num)]
        for i in range(child_num):
            self.children[i] = Child(
                lock=lock,
                send_queue=send_s_pipe,
                recv_queue=recv_res_pipes[i],
                states_list=states_list,  #
                child_id=i + 1  # child_id  from 1 to child_num
            )


# ------------------------------------------------------
#                        test
# -------------------------------------------------------


def master_f(master: Master):
    brain_memory = []
    while True:
        try:
            child_id, data = master.recv()
            flag, origin_data = data
        except:
            child_id = -1
            data = None
        if child_id == -1:
            if all(master.states_list[1:]):
                break
        else:
            if flag:
                master.send(origin_data + 100, child_id)
            else:
                brain_memory.append(origin_data)

                # tell child can change his state
                master.send("ok", child_id)

    if len(brain_memory) == process_num:
        print("master           :   OK")


# data structure
# [flag , origin data]
# when flag = 1 , mean agent is playing   , this time data is frame
# when flag = 0 , mean agent had finished , this time data is all memory

def child_f(child: Child):
    for i in range(child.child_id):
        child.send([1, i])
        res = child.recv()
        if res != i + 100:
            print("error in child_f")

    child.send([0, "big memory"])
    if child.recv() == "ok":
        child.set_state(1)
    else:
        print("child_f_" + str(child.child_id) + "        :   ERROR")
    print("child_f_" + str(child.child_id) + "        :   OK")


def main():
    communication = Communication(child_num=process_num)
    master_p = Process(target=master_f, args=(communication.master,))
    child_p = []
    for i in range(process_num):
        child_p.append(Process(target=child_f, args=(communication.children[i],)))
    master_p.start()
    for p in child_p:
        p.start()
    master_p.join()
    for p in child_p:
        p.join()


if __name__ == '__main__':
    main()
