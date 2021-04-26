from environment import Env
from tools import Tools
from DQN import *
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import math

if __name__ == '__main__':
    # 实例化
    n = 2
    env = Env()
    tools = Tools()
    with tf.Session() as sess:
        rl = DQN(
            sess=sess,
            s_dim=3 * n,
            a_dim=int(math.pow(env.range,n)),
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
        # # 实时路况图
        # color_table = ['k', 'b', 'r', 'y', 'g', 'c', 'm']
        # txt_table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # reward图
        epi = []
        success = []


        for episode in range(10000):
            print('episode',episode)
            epi.append(episode)

            total = 0
            suss = 0

            state = env.reset(n, tools)

            while True:

                temp_state = tools.get_list(state)  # 车组中所有车辆状态合成
                add_action = rl.choose_action(np.array(temp_state))    # 学习到车组的动作组合
                add_action1 = rl.choose_action(np.array(temp_state))    # 学习到车组的动作组合


                # 车组动作组合转换成车辆的单个动作增量
                add = []
                b = []
                for k in range(n):
                    s = add_action1 // env.range  # 商
                    y = add_action1 % env.range  # 余数
                    b = b + [y]
                    add_action1 = s
                b.reverse()
                for i in b:
                    add.append(i)

                # 转换成车辆的单个动作
                act = []
                for dim in range(n):
                    act.append(state[dim][1] - env.range / 2 + add[dim])

            # print('更新')
                state_, reward, done, draw = env.step(act,state,n)  # dicreward改成一个值
            # unchange_dic_state_:没进行车辆更新之前的车辆下一时刻的状态
            # draw_pos:用于画图，记录（位置更新）之前的车辆位置
            # draw_act:用于画图，记录（位置更新）之前的车辆动作

            # 画图

                plt.sca(ax1)
                ax1.cla()  # 清空画布
                plt.axis([0, 200, 0, 0.1])  # 坐标轴范围
                # x_major_locator = MultipleLocator(5)  # 把x轴的刻度间隔设置为1，并存在变量里
                # plt.gca()  # ax为两条坐标轴的实例
                # plt.xaxis.set_major_locator(MultipleLocator(5))  # 把x轴的主刻度设置为1的倍数
                # figure1.tick_params(axis='both', which='major', labelsize=5)  # 坐标轴字体大小
                y1 = []
                y2 = []
                pos_x = []
                pos_y = []
                for t in range(n):
                    pos_x.append(draw[t])
                    pos_y.append(state[t][0]* env.per_section)
                    y1.append(0)
                    y2.append(0.02)

                plt.scatter(pos_x, y1, marker="o")  # 画图数据
                plt.scatter(pos_y, y2, marker="o")  # 画图数据
                plt.pause(env.frame_slot)

                l_temp_state = tools.get_list(state)
                l_temp_state_ = tools.get_list(state_)
                rl.store_transition_and_learn(l_temp_state, add_action, reward, l_temp_state_, done)

                total += env.data_beam_slot+env.choose_beam_slot
                suss += reward

                state = state_
                if done:
                    break

            plt.sca(ax2)
            ax2.cla()
            # plt.ylim([0, 5000])  # 坐标轴范围
            success.append(suss/total)
            plt.plot(epi, success)
            plt.pause(env.frame_slot)

        # print('成功率',suss/total)






