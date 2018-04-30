import tensorflow as tf
import numpy as np

# hyper parameters

LR_A = 0.001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 30000
BATCH_SIZE = 50


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, ):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1),
                               dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]

        # placeholders
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        print "Building Graph..."
        self._build_graph()
        print "Graph Built"

    def choose_action(self, s):
        return self.sess.run(self.a, feed_dict={
            self.S: s[None, :]
        })[0]

    def learn(self):
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)

        # self.memory : [s, a, reward, s_]
        batch = self.memory[indices, :]
        batch_s = batch[:, :self.s_dim]
        batch_a = batch[:, self.s_dim:self.s_dim + self.a_dim]
        batch_r = batch[:, -self.s_dim - 1:-self.s_dim]
        batch_s_ = batch[:, -self.s_dim:]

        self.sess.run(self.a_train, feed_dict={self.S: batch_s})
        self.sess.run(self.c_train, feed_dict={self.S: batch_s,
                                               self.a: batch_a,
                                               self.R: batch_r,
                                               self.S_: batch_s_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:
            self.memory_full = True

    def _build_graph(self):
        with tf.variable_scope('Actor'):
            self.a = self.actor(self.S, scope='eval', trainable=True)
            a_ = self.actor(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error
            # otherwise the self.a is from Actor when updating Actor
            q = self.critic(self.S, self.a, scope='eval', trainable=True)
            q_ = self.critic(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(at, (1 - TAU) * at + TAU * ae),
                              tf.assign(ct, (1 - TAU) * ct + TAU * ce)
                              ]
                             for at, ae, ct, ce in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.c_train = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.a_train = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

        tf.summary.FileWriter("logs/", self.sess.graph)

    def actor(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 300, activation=tf.nn.relu,
                                 name='l1', trainable=trainable)
            a = tf.layers.dense(l1, self.a_dim, activation=tf.nn.tanh,
                                name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def critic(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            w1_s = tf.get_variable('w1_s', [self.s_dim, 300], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, 300], trainable=trainable)
            b1 = tf.get_variable('b1', [1, 300], trainable=trainable)
            l1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1, name='l1')
            l2 = tf.layers.dense(l1, 1, name='l2', trainable=trainable)
            return l2

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './model/ddpg.ckpt')

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './model/ddpg.ckpt')
