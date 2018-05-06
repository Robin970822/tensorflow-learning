# coding=utf-8
# 导入环境和学习方法
from arm_env import ArmEnv
from car_env import CarEnv
from rl import DDPG

# 设置全局变量
MAX_EPISODES = 500
MAX_EP_STEPS = 300
ON_TRAIN = True


def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            if i >= 200:
                env.render()
            a = rl.choose_action(s)
            s_, r, done = env.step(a)
            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                # print 'Learning...'
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS - 1:
                print 'Episode: %i | %s |Reward: %.1f | Step: %i' % (i, '---' if not done else 'done', ep_r, j + 1)
                break
    rl.save()


def eval():
    rl.restore()
    env.reset()
    s = env.reset()
    while True:
        env.render()
        a = rl.choose_action(s)
        env.step(a)


if __name__ == '__main__':

    # 设置环境
    env = ArmEnv()
    s_dim = env.state_dim
    a_dim = env.action_dim
    a_bound = env.action_bound

    # 设置学习方法
    rl = DDPG(a_dim, s_dim, a_bound)

    # 开始训练
    if ON_TRAIN:
        train()
    else:
        eval()
