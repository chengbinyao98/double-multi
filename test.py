import matplotlib.pyplot as plt
import numpy as np
import math

road_length = 200
straight = 100
ann_num = 16
dis = 0.5

pos = 105

act = []
SNR = []

for i in range(int(road_length/dis)):

    a = abs(road_length / 2 - pos)
    # 斜边
    b = np.sqrt(np.square(a) + np.square(straight))
    if pos > road_length / 2:
        th1 = math.pi - math.acos(a / b)
    else:
        th1 = math.acos(a / b)
    print('th1',math.degrees(th1))

    # channel = []
    # for t in range(ann_num):
    #     m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
    #     channel.append(m.conjugate())
    # print('channel',channel)


    act.append(dis * i)
    # 直角边
    c = abs(road_length / 2 - act[i] )
    print('c',c)
    # 斜边
    d = np.sqrt(np.square(c) + np.square(straight))
    if act[i] > road_length / 2:
        th2 = math.pi - math.acos(c / d)
    else:
        th2 = math.acos(c / d)
    print('th2', math.degrees(th2))

    channel = []
    for t in range(ann_num):
        m = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
        channel.append(m.conjugate())
    print('channel',channel)

    signal = []
    for t in range(ann_num):
        n = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
        signal.append(n)
    print('signal',signal)

    # print('dot',np.dot(channel, signal))
    # print('norm', np.linalg.norm(np.dot(channel, signal)))
    # print('square',np.square(np.linalg.norm(np.dot(channel, signal))))

    # SNR.append(10 * math.log10(np.square(np.linalg.norm(np.dot(channel, signal)))))
    SNR.append(np.square(np.linalg.norm(np.dot(channel, signal))))

plt.scatter(act, SNR)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.show()