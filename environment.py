import numpy as np
import math
from tools import Tools
#
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


class Env(object):
    def __init__(self):

        # 帧结构
        self.frame_slot = 0.01          # 帧时隙时间长度
        self.choose_beam_slot = 30       # 波束选择时隙数
        self.data_beam_slot = 70         # 数据传输时隙数
        self.right = 200                 # 正确传输最低的SNR

        self.frame_period = (self.choose_beam_slot + self.data_beam_slot) * self.frame_slot  # 帧周期时间长度

        # 车辆和道路
        '''
        self.k = 4    # 车辆进行一次波束变换最好可以支撑k个帧周期
        为了波束可以支撑k个帧周期，当前车辆的动作选择范围应该和k个帧周期之后的动作选择范围有重叠:
        v * self.k * (self.data_beam_slot + self.choose_beam_slot) * self.frame_slot < self.road_range  v < 10
        
        self.T = 40   # 车辆平均在T个帧周期走完全程
        可以求得车辆的平均配速:v = self.road_length / self.T / self.frame_period = 10
        
        self.M = 3  # 期望大约最多有几辆车成为一个车辆簇
        车辆之间的初始距离如果大于车辆之间的干扰距离，那么如果后车的速度大于前车的速度，前车在加速的过程中必定拉近两车间距，变成有干扰的距离,
        这里取极端情况，假设M辆车的间距刚好是开始反应的距离两端的车速是最大和最小，则两端的车辆共速时候车辆间距将减少:
        math.pow((self.v_max - self.v_min), 2) / 2 / self.accelerate
        如果保证可以有M个车辆可以产生干扰，则:
        self.min_dis - math.pow((self.v_max - self.v_min), 2) / 2 / self.accelerate / self.M < self.no_interference
        如果小于M个车辆，两端速度不是最大最小时候也可以，如果大于M个车辆，由于上式是小于非等于，所以也有可能，而且往前走车辆堆积或许可以产生此种情况
        
        为了在干扰范围内车辆之间可以产生干扰，干扰距离之内的两车动作选择范围应该有重叠，而且重叠范围不能太小也不能太大，这里取一半
        self.no_interference  =  self.road_range / 2 = 10
        
        根据车辆的平均配速和一个周期的时间，路段的划分应该能体现车辆的移动，同时为了保证状态和动作的数目不太多，不能设置太小
        '''

        # 车辆和道路
        self.road_length = 200  # 道路长度
        self.straight = 100  # 基站和道路的直线距离

        # self.no_interference = 15      # 车辆没有干扰的距离
        self.no_interference = 20  # 车辆没有干扰的距离

        self.v_min = 8  # 车辆的最小速度
        self.v_max = 16  # 车辆的最大速度
        self.accelerate = 16  # 车辆的加速度
        self.min_dis = 22  # 车辆之间的最小反应距离
        self.max_dis = 28

        self.per_section = 2  # 每几米划分成一个路段

        # self.l_min = 10                 # 车辆间距服从均匀分布
        # self.l_max = 12

        # 天线
        self.ann_num = 16  # 天线数目
        self.road_range = 50  # 动作可以选择的范围
        self.range = self.road_range / self.per_section  # 动作选择范围

        # 存储单元
        self.cars_posit = []  # 车辆的位置（连续）
        self.cars_speed = []  # 车辆的速度（连续
        # self.action = []                # 一个帧周期的动作     ？？？？？

    # 由道路上的所有车辆得到所有车辆的路段
    def get_section(self, list):
        section = []
        for i in range(len(list)):
            section.append(math.ceil(list[i] / self.per_section))
        return section

    def road_reset(self,n):
        self.cars_posit = []  # 车辆的位置（连续）
        self.cars_speed = []  # 车辆的速度（连续
        for i in range(n):
            self.cars_speed.append(np.random.uniform(self.v_min, self.v_max))
            if i == 0:
                self.cars_posit.append(0)
            else:
                self.cars_posit.append(self.cars_posit[i-1] + np.random.uniform(self.min_dis,self.max_dis))

    # 道路路面现有车辆更新
    def road_step(self):
        mark = 0  # 标记当前车辆是否之前被操作过，保证一个时隙车只跑一个时隙的量
        for i in range(len(self.cars_posit) - 1):
            if mark == 0:
                if self.cars_posit[i + 1] - self.cars_posit[i] < self.min_dis:
                    if np.random.rand() < 0.5:
                        cars_speed_next = self.cars_speed[i] - self.accelerate * self.frame_slot
                        # 减速到最小速度即可
                        if cars_speed_next <= self.v_min:
                            cars_speed_next = self.v_min
                        ti = (self.cars_speed[i] - cars_speed_next) / self.accelerate
                        self.cars_posit[i] = self.cars_speed[i] * ti - ti * ti * self.accelerate / 2 + (
                                self.frame_slot - ti) * cars_speed_next + self.cars_posit[i]
                        self.cars_speed[i] = cars_speed_next
                        mark = 0
                    else:
                        cars_speed_next = self.cars_speed[i + 1] + self.accelerate * self.frame_slot
                        # 减速到最小速度即可
                        if cars_speed_next >= self.v_max:
                            cars_speed_next = self.v_max
                        ti1 = (cars_speed_next - self.cars_speed[i + 1]) / self.accelerate
                        self.cars_posit[i + 1] = self.cars_speed[i + 1] * ti1 + ti1 * ti1 * self.accelerate / 2 + (
                                self.frame_slot - ti1) * cars_speed_next + self.cars_posit[i + 1]
                        self.cars_speed[i + 1] = cars_speed_next
                        self.cars_posit[i] = self.cars_speed[i] * self.frame_slot + self.cars_posit[i]
                        mark = 1
                if self.cars_posit[i + 1] - self.cars_posit[i] > self.max_dis:
                    if np.random.rand() < 0.5:
                        cars_speed_next = self.cars_speed[i+1] - self.accelerate * self.frame_slot
                        # 减速到最小速度即可
                        if cars_speed_next <= self.v_min:
                            cars_speed_next = self.v_min
                        ti1 = (self.cars_speed[i+1] - cars_speed_next) / self.accelerate
                        self.cars_posit[i+1] = self.cars_speed[i+1] * ti1 - ti1 * ti1 * self.accelerate / 2 + (
                                self.frame_slot - ti1) * cars_speed_next + self.cars_posit[i+1]
                        self.cars_speed[i+1] = cars_speed_next
                        self.cars_posit[i] = self.cars_speed[i] * self.frame_slot + self.cars_posit[i]
                        mark = 2
                    else:
                        cars_speed_next = self.cars_speed[i] + self.accelerate * self.frame_slot
                        # 减速到最小速度即可
                        if cars_speed_next >= self.v_max:
                            cars_speed_next = self.v_max
                        ti = (cars_speed_next - self.cars_speed[i]) / self.accelerate
                        self.cars_posit[i] = self.cars_speed[i] * ti + ti * ti * self.accelerate / 2 + (
                                self.frame_slot - ti) * cars_speed_next + self.cars_posit[i]
                        self.cars_speed[i] = cars_speed_next
                        mark = 0
                if self.min_dis < self.cars_posit[i + 1] - self.cars_posit[i] < self.max_dis:
                    self.cars_posit[i] = self.cars_speed[i] * self.frame_slot + self.cars_posit[i]
                    mark = 0
            else:
                if self.cars_posit[i + 1] - self.cars_posit[i] < self.min_dis:
                    cars_speed_next = self.cars_speed[i + 1] + self.accelerate * self.frame_slot
                    # 减速到最小速度即可
                    if cars_speed_next >= self.v_max:
                        cars_speed_next = self.v_max
                    ti1 = (cars_speed_next - self.cars_speed[i + 1]) / self.accelerate
                    self.cars_posit[i + 1] = self.cars_speed[i + 1] * ti1 + ti1 * ti1 * self.accelerate / 2 + (
                            self.frame_slot - ti1) * cars_speed_next + self.cars_posit[i + 1]
                    self.cars_speed[i + 1] = cars_speed_next
                    mark = 1
                if self.cars_posit[i + 1] - self.cars_posit[i] > self.max_dis:
                    cars_speed_next = self.cars_speed[i+1] - self.accelerate * self.frame_slot
                    # 减速到最小速度即可
                    if cars_speed_next <= self.v_min:
                        cars_speed_next = self.v_min
                    ti1 = (self.cars_speed[i+1] - cars_speed_next) / self.accelerate
                    self.cars_posit[i+1] = self.cars_speed[i+1] * ti1 - ti1 * ti1 * self.accelerate / 2 + (
                            self.frame_slot - ti1) * cars_speed_next + self.cars_posit[i+1]
                    self.cars_speed[i+1] = cars_speed_next
                    mark = 2
                if self.min_dis < self.cars_posit[i + 1] - self.cars_posit[i] < self.max_dis:
                    mark = 0
        if mark == 0:
            self.cars_posit[len(self.cars_posit) - 1] = self.cars_speed[len(self.cars_posit) - 1] * self.frame_slot + \
                                                        self.cars_posit[len(self.cars_posit) - 1]

    # 路面上新进入车辆或者有车辆驶出路段进行更新
    # def get_information(self, action, section, n):
    #     for i in range(10):
    #         # 将超出道路的车辆排除
    #         if self.cars_posit[len(self.cars_posit) - 1] > self.road_length:
    #             for j in range(n):
    #                 del action[len(self.cars_posit) - 1]
    #                 del section[len(self.cars_posit) - 1]
    #                 del self.cars_speed[len(self.cars_posit) - 1]
    #                 del self.cars_posit[len(self.cars_posit) - 1]
    #         else:
    #             break
    #     return action, section

    def choose_reward(self, act,reward,n):
        for i in range(n):
            SNR_noise = 0
            SNR = 0
            for j in range(n):
                # 直角边
                a = abs(self.road_length / 2 - self.cars_posit[i])
                # 斜边
                b = np.sqrt(np.square(a) + np.square(self.straight))
                if self.cars_posit[i] > self.road_length / 2:
                    th1 = math.pi - math.acos(a / b)
                else:
                    th1 = math.acos(a / b)

                channel = []
                for t in range(self.ann_num):
                    m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
                    channel.append(m.conjugate())

                # 直角边
                c = abs(self.road_length / 2 - act[j] * self.per_section)
                # 斜边
                d = np.sqrt(np.square(c) + np.square(self.straight))
                if act[j] * self.per_section > self.road_length / 2:
                    th2 = math.pi - math.acos(c / d)
                else:
                    th2 = math.acos(c / d)

                signal = []
                for t in range(self.ann_num):
                    p = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
                    signal.append(p)

                if j != i:
                    SNR_noise += np.square(np.linalg.norm(np.dot(channel, signal)))
                else:
                    SNR = np.square(np.linalg.norm(np.dot(channel, signal)))
            if SNR_noise == 0:
                if SNR >= self.right:
                    reward[i] += 1
            else:
                if SNR / SNR_noise >= self.right:
                    reward[i] += 1
        return reward


    # def data_reward(self, act, reward):
    #     # 直角边
    #     a = abs(self.road_length / 2 - self.cars_posit)
    #     # 斜边
    #     b = np.sqrt(np.square(a) + np.square(self.straight))
    #     if self.cars_posit > self.road_length / 2:
    #         th1 = math.pi - math.acos(a / b)
    #     else:
    #         th1 = math.acos(a / b)
    #
    #     channel = []
    #     for t in range(self.ann_num):
    #         m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
    #         channel.append(m.conjugate())
    #
    #     # 直角边
    #     c = abs(self.road_length / 2 - act * self.per_section)
    #     # 斜边
    #     d = np.sqrt(np.square(c) + np.square(self.straight))
    #     if act * self.per_section > self.road_length / 2:
    #         th2 = math.pi - math.acos(c / d)
    #     else:
    #         th2 = math.acos(c / d)
    #
    #     signal = []
    #     for t in range(self.ann_num):
    #         n = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
    #         signal.append(n)
    #
    #     SNR = np.square(np.linalg.norm(np.dot(channel, signal)))
    #
    #     if SNR >= self.right:
    #         self.reward[0] += 1
    #         reward += 1
    #
    #     self.reward[1] += 1
    #     return reward

    def reset(self,n,tool):
        # 道路环境初始化
        self.road_reset(n)

        _section = self.get_section(self.cars_posit)          # 前一时刻的路段
        beam = self.get_section(self.cars_posit)           # 前一时刻的动作

        # # 初始化每个车辆的回报
        # self.reward = [0, 0]

        for i in range(self.data_beam_slot+self.choose_beam_slot):
            self.road_step()
        now = self.get_section(self.cars_posit)
        state = []
        for i in range(n):
            state.append([beam[i],now[i],_section[i]])
        return state


    def step(self, action, state,n):

        _section = self.get_section(self.cars_posit)                    # 当前车辆的路段  用于产生下一时刻的状态
        draw_pos = []  # 记录当前道路的位置，用于画图
        for i in range(len(self.cars_posit)):
            draw_pos.append(self.cars_posit[i])

        # 道路的（位置更新）
        reward = [0 for i in range(n)]
        for i in range(self.choose_beam_slot+self.data_beam_slot):
            reward = self.choose_reward(action,reward,n)
            self.road_step()
        # print('choose', reward)
        # for i in range(self.data_beam_slot):
        #     reward = self.data_reward(action,reward)
        #     self.road_step()
        # print('data',reward)
        reward1 = 0
        for i in range(n):
            reward1 += reward[i]

        now = self.get_section(self.cars_posit)
        state_ = []
        for i in range(n):
            state_.append([action[i],now[i],_section[i]])

        if self.cars_posit[n-1] > self.road_length or self.cars_posit[0] > self.road_length:
            done = 1
        else:
            done = 0

        return state_,reward1,done,draw_pos



    def draw(self):

        plt.ion()  # 开启交互模式
        plt.figure(figsize=(100, 3))    #  设置画布大小


        # 数据
        y = []
        for i in range(len(self.cars_posit)):
            y.append(0)

        for j in range(1000):
            plt.clf()  # 清空画布
            plt.axis([0, 210, 0, 0.1])  # 坐标轴范围
            x_major_locator = MultipleLocator(5)     # 把x轴的刻度间隔设置为1，并存在变量里
            ax = plt.gca()    # ax为两条坐标轴的实例
            ax.xaxis.set_major_locator(x_major_locator)   # 把x轴的主刻度设置为1的倍数
            plt.tick_params(axis='both', which='major', labelsize=5)    # 坐标轴字体大小

            self.road_step()

            plt.scatter(self.cars_posit, y, marker="o")    # 画图数据
            plt.pause(0.2)

        plt.ioff()
        plt.show()




if __name__ == '__main__':
    env = Env()
    tools = Tools()
    dic_state = env.reset(tools)
    # # dic_state = env.reset(tools)
    # print('车辆位置')
    # print(env.cars_posit)
    # print('车辆地段')
    # print(env.cars_section)
    # print('状态')
    # print(dic_state)
    # print('车辆信息')
    # print(tools.cars_info)
    #
    # dic_action = {}
    # for x in dic_state:
    #     if x not in dic_action:
    #         dic_action[x] = []
    #     for i in range(len(dic_state[x])):
    #         act = [np.random.randint(low=1, high=50) for j in range(len(dic_state[x][i]))]
    #         dic_action[x].append(act)
    # print('动作')
    # print(dic_action)
    #
    # dic_state_, dic_reward, done = env.step(dic_action,dic_state,tools)
    #
    #
    # print('车辆位置')
    # print(env.cars_posit)
    # print('车辆地段')
    # print(env.cars_section)
    # print('车辆信息')
    # print(tools.cars_info)
    # print('下一状态')
    # print(dic_state_)
    # print('回报')
    # print(dic_reward)
    # print('完成')
    # print(done)
    env.draw()
