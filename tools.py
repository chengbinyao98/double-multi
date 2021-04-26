# from environment import Env
import numpy as np

class Tools(object):


    def get_info(self,pos,no_interference):  # 看看不用info的内部存储可以么
        dim = 0
        num = {}
        cars_info = [[0 for m in range(3)] for p in range(len(pos))]    #  注意以后再用的时候能不能行
        for i in range(len(pos)):
            dim += 1
            if i != len(pos) - 1:
                if pos[i + 1] - pos[i] > no_interference:
                    if dim not in num:
                        num[dim] = 0  # 从0开始计算组数
                    else:
                        num[dim] += 1
                    for p in range(i + 1 - dim, i + 1):
                        cars_info[p][0] = dim
                        cars_info[p][1] = num[dim]
                        cars_info[p][2] = p - i - 1 + dim
                    dim = 0
            else:
                if dim not in num:
                    num[dim] = 0  # 从0开始计算组数
                else:
                    num[dim] += 1
                for p in range(i + 1 - dim, i + 1):
                    cars_info[p][0] = dim
                    cars_info[p][1] = num[dim]
                    cars_info[p][2] = p - i - 1 + dim
        return cars_info

    def get_list(self, a):
        temp_state = []
        for dim in range(len(a)):
            temp_state.append(a[dim][0])
            temp_state.append(a[dim][1])
            temp_state.append(a[dim][2])
        temp_state = np.array(temp_state)
        return temp_state

    def reverse_classify(self, dic, info):
        a = []
        for i in range(len(info)):
            dim = info[i][0]
            num = info[i][1]
            number = info[i][2]
            a.append(dic[dim][num][number])
        return a

    def classify(self, list_one, info):
        a = {}
        # 逐个车辆进行分类
        for i in range(len(list_one)):
            # 如果此类型车辆为新类型，创建一个空列表
            if info[i][0] not in a:
                a[info[i][0]] = []
                # 如果该类型新产生一组，则增加一个n智能体的0列表
                if info[i][1] + 1 > len(a[info[i][0]]):
                    a[info[i][0]].append([0 for k in range(info[i][0])])
                    a[info[i][0]][info[i][1]][info[i][2]] = list_one[i]
                else:
                    a[info[i][0]][info[i][1]][info[i][2]] = list_one[i]
            else:
                # 如果该类型新产生一组，则增加一个n智能体的0列表
                if info[i][1] + 1 > len(a[info[i][0]]):
                    a[info[i][0]].append([0 for k in range(info[i][0])])
                    a[info[i][0]][info[i][1]][info[i][2]] = list_one[i]
                else:
                    a[info[i][0]][info[i][1]][info[i][2]] = list_one[i]
        return a

    def integrate(self,dic_a,dic_b,dic_c):
        a = {}
        for x in dic_a:
            if x not in a:
                a[x] = [[[0 for m in range(3)] for p in range(len(dic_a[x][0]))] for q in range(len(dic_a[x]))]
            for i in range(len(dic_a[x])):
                for j in range(len(dic_a[x][i])):
                    a[x][i][j][0] = dic_a[x][i][j]
                    a[x][i][j][1] = dic_b[x][i][j]
                    a[x][i][j][2] = dic_c[x][i][j]
        return a



# if __name__ == '__main__':
#     env = Env()
#     tools = Tools()
#     env.road_reset()
#     print(env.cars_posit)
#     tools.get_info(env.cars_posit,env.no_interference)
#     print(tools.cars_info)
#     a = tools.classify(env.cars_posit)
#     print(a)
#     b = tools.integrate(a,a,a)
#     print(b)







