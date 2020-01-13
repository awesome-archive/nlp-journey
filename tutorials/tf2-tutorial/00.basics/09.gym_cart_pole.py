# coding=utf-8
# created by msg on 2019/11/23 10:11 上午

import gym

"""
一个简单的游戏： CartPole
"""
env = gym.make('CartPole-v0')

env.reset()
for _ in range(1000):
    env.render()
    env.step(action=env.action_space.sample())
env.close()
