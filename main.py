from environment import Env
from tools import Tools
from DQN import *
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import math
from Draw import DRAW


if __name__ == '__main__':
    # 实例化
    n = 2
    env = Env()
    tools = Tools()
    draw = DRAW()

    with tf.Session() as sess:
        rl = DQN(
            sess=sess,
            s_dim=3 * n,
            a_dim=int(math.pow(env.road_range,n)),
            batch_size=128,
            gamma=0.99,
            lr=0.01,
            epsilon=0.1,
            replace_target_iter=300
        )
        tf.global_variables_initializer().run()


        # 画图
        plt.ion()
        plt.figure(figsize=(100, 5))    # 设置画布大小
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        # reward图
        epi = []
        success = []


        for episode in range(10000):
            print('episode',episode)
            epi.append(episode)

            total_reward = 0
            time = 0

            state = env.reset(n, tools)
            done = 0

            while True:

                temp_state = tools.get_list(state)  # 车组中所有车辆状态合成
                add_action = rl.choose_action(np.array(temp_state))    # 学习到车组的动作组合
                add_action1 = rl.choose_action(np.array(temp_state))    # 学习到车组的动作组合


                # 车组动作组合转换成车辆的单个动作增量
                add = []
                b = []
                for k in range(n):
                    s = add_action1 // env.road_range  # 商
                    y = add_action1 % env.road_range  # 余数
                    b = b + [y]
                    add_action1 = s
                b.reverse()
                for i in b:
                    add.append(i)

                # 转换成车辆的单个动作
                action = []
                for dim in range(n):
                    action.append(int(env.cars_posit[dim]) - env.road_range / 2 + add[dim])

                draw.piant(env.cars_posit, env.road_range, ax1, env.frame_slot, n, action)

                state_, reward, done = env.step(action,state,n)  # dicreward改成一个值

                l_temp_state = tools.get_list(state)
                l_temp_state_ = tools.get_list(state_)
                rl.store_transition_and_learn(l_temp_state, add_action, reward, l_temp_state_, done)

                total_reward += reward
                time += 1

                state = state_
                if done:
                    break

            plt.sca(ax2)
            ax2.cla()
            success.append(total_reward/(env.beam_slot*time*n))
            plt.plot(epi, success)
            plt.pause(env.frame_slot)







