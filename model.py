import time
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
import numpy as np
from config import *
from datetime import datetime

logdir = "./logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
weight_dir = "./logs/weight/"
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
index = 0
if index == 0:
    restore = False
else:
    restore = True
restore_weight_dir = "./logs/weight/{}".format(index)


class ACModel(Model):
    def __init__(self):
        super(ACModel, self).__init__()
        self.c1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                         activation='relu')
        self.c2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu')
        self.c3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(512, activation="relu")
        self.d2 = Dense(1)  # C
        self.d3 = Dense(a_num, activation='softmax')  # A
        self.save_index = index
        self.total_index = index
        self.call(np.random.random((batch_size, IMG_H, IMG_W, k)).astype(np.float32))
        if restore:
            self.load_weights(restore_weight_dir)

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

    def loss(self, inputs, targets, action_index, adv, ep_old_ap):
        res = self.call(inputs)
        error = res[1][:, 0] - targets
        L = tf.reduce_sum(tf.square(error))

        adv = tf.dtypes.cast(tf.stop_gradient(adv), tf.float32)
        batch_size = inputs.shape[0]
        all_act_prob = res[0]
        selected_prob = tf.reduce_sum(action_index * all_act_prob, axis=1)
        old_prob = tf.reduce_sum(action_index * ep_old_ap, axis=1)

        r = selected_prob / (old_prob + 1e-6)

        H = -tf.reduce_sum(all_act_prob * tf.math.log(all_act_prob + 1e-6))

        Lclip = tf.reduce_sum(
            tf.minimum(
                tf.multiply(r, adv),
                tf.multiply(
                    tf.clip_by_value(
                        r,
                        1 - clip_epsilon,
                        1 + clip_epsilon
                    ),
                    adv
                )
            )
        )

        return -(Lclip - VFcoeff * L + beta * H) / batch_size, Lclip, H, L

    def total_grad(self, ep_stack_obs, ep_as, adv, realv, ep_old_ap):
        with tf.GradientTape() as tape:
            loss_value, Lclip, H, L = self.loss(ep_stack_obs, realv, ep_as, adv, ep_old_ap)
            self.total_index += 1

            if self.total_index % recode_span == 1:
                self.record('total loss', loss_value * len(ep_as))
                self.record('Lclip', Lclip)
                self.record('H', H)
                self.record('c loss', L)

            self.save_index += 1
            if self.save_index % save_span == 0:
                self.save_weights(weight_dir + str(self.save_index), save_format='tf')
        return tape.gradient(loss_value, self.trainable_weights), loss_value

    def record(self, name, data, step=None):
        if not step:
            step = self.total_index
        tf.summary.scalar(name, data=data, step=step)


def test1():
    m = ACModel()
    m.build((None, IMG_H, IMG_W, k))
    m.summary()

    inputs = np.random.random((1, IMG_H, IMG_W, k)).astype(np.float32)
    a, c = m(inputs)

    inputs = np.random.random((1, IMG_H, IMG_W, k)).astype(np.float32)
    s = time.time()
    a, c = m(inputs)
    print(time.time() - s)

    inputs = np.random.random((32, IMG_H, IMG_W, k)).astype(np.float32)
    s = time.time()
    a, c = m(inputs)
    print(time.time() - s)


if __name__ == '__main__':
    test1()
