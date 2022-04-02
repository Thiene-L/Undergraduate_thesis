import random
import numpy as np
import pandas as pd
from scipy.integrate import odeint

epsilon = 0.9  # 贪婪度 greedy
learning_alpha = 0.1  # 学习率
gamma = 0.8  # 奖励递减值
alpha, beta = 1, -3  # 定义alpha和beta值
a = 3.  # 定义a的值
T = 1.  # 定义T的值
step = 0.01  # 定义时间t步长
times = T / step - 1  # 定义步数
u_step = 1  # 定义u取值的步长
u_min = -10.  # 定义u的最小值
u_max = 10.  # 定义u的最大值
u_num = (u_max - u_min) / u_step  # 定义u的取值个数
states = np.arange(step, T, step)  # 定义探索状态，根据时间t来划分
actions = np.arange(u_min, u_max + u_step, u_step)  # 定义动作集
rewards = np.zeros(int(times), dtype=float)  # 回报值就是目标函数的值，目前初始化全部为0
learningTimes = 100  # 定义学习次数
states1 = np.arange(int(times))
# 定义q_table
q_table = pd.DataFrame(data=[[0. for _ in actions] for _ in states],
                       index=states1, columns=actions)


# 求解确定t时刻，x的解
def xt(alpha, beta, u):
    def func(x, t):
        dxdt = alpha * x + beta * u
        return dxdt

    t = np.arange(step, T, step)
    return odeint(func, 0., t)


# 确定目标函数的值
def cu(a, xt, u):
    return (a * (xt ** 2) + u ** 2) / 2


# 对状态执行动作后，得到下一个状态
def get_next_state(state):
    return state + 1


# 取当前状态下的合法动作集合，与reward无关
def get_valid_actions():
    global actions
    valid_actions = set(actions)
    return list(valid_actions)


# 学习learningTimes的次数
for i in range(learningTimes):
    current_state = 0

    while current_state != int(states[-1] / step - 1):
        # 开始探索
        if (random.uniform(0, 1) > epsilon) or (q_table.loc[current_state].abs() < 1e-6).all():
            current_action = random.choice(get_valid_actions())
        else:
            # 利用（贪婪）
            current_action = q_table.loc[current_state].idxmax()

        # 获取下一个状态
        next_state = get_next_state(current_state)
        # 获取此时的xt值
        x = xt(alpha, beta, current_action)[current_state][0]
        # 更新rewards，当前的reward为cu的值
        rewards[next_state] = -cu(a, x, current_action)
        # 下一个状态的q_table
        next_state_q_values = q_table.loc[next_state]
        # 更新q_table
        q_table.loc[current_state, current_action] += learning_alpha * (
                rewards[next_state] + gamma * next_state_q_values.max() - q_table.loc[
            current_state, current_action])
        current_state = next_state

    # 清空rewards
    rewards = np.zeros(int(times), dtype=float)

