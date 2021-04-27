import matplotlib.pyplot as plt
import numpy as np
import math

class DRAW(object):

    def piant(self,pos,road_range,ax,frame_slot,n, action):
        road_length = 200
        straight = 100
        ann_num = 16

        plt.sca(ax)
        ax.cla()
        plt.axis([0, 210, 0, 270])  # 坐标轴范围

        y1 = [0 for i in range(n)]
        y2 = [50 for i in range(n)]
        plt.scatter(pos, y1, marker="o")  # 画图数据
        plt.scatter(action, y2, marker="o")  # 画图数据

        for j in range(len(pos)):
            SNR = []
            act = []
            for i in range(road_range):

                a = abs(road_length / 2 - pos[j])
                # 斜边
                b = np.sqrt(np.square(a) + np.square(straight))
                if pos[j] > road_length / 2:
                    th1 = math.pi - math.acos(a / b)
                else:
                    th1 = math.acos(a / b)
                channel = []
                for t in range(ann_num):
                    m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
                    channel.append(m.conjugate())

                act.append(int(pos[j]) - road_range/2 + i)
                # 直角边
                c = abs(road_length / 2 - act[i] )
                # 斜边
                d = np.sqrt(np.square(c) + np.square(straight))
                if act[i] > road_length / 2:
                    th2 = math.pi - math.acos(c / d)
                else:
                    th2 = math.acos(c / d)
                signal = []
                for t in range(ann_num):
                    n = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
                    signal.append(n)

                SNR.append(np.square(np.linalg.norm(np.dot(channel,signal))))
            plt.plot(act, SNR)

        plt.pause(frame_slot)
