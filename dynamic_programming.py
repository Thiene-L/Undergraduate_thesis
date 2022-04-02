# 利用近似动态规划方法计算扩散方程最优控制问题
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt

# MacOS系统比较特殊，这一行是为了让MacOS系统能够显示中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 设置参数与读入数据
domain = 1  # x的范围
h = 0.02  # x的步长
time = 1  # 时间t的范围
dt = 0.000025  # 时间步长

# 建立x矩阵
x = np.arange(0, domain + h, h)
# 建立t矩阵
t = np.arange(0, time + dt, dt)
# 定义x矩阵长度为M
M = len(x)
# 定义时间步数
n_iters = len(t) - 1


# 定义Laplacian矩阵
def laplace_matrix(order):
    global h
    vec1 = np.ones(order)
    vec2 = -1 * np.ones(order)
    vec2[1:len(vec2) - 1] = 2 * vec2[1:len(vec2) - 1]
    func = 1 / h ** 2 * sparse.spdiags(np.array([vec1, vec2, vec1]), np.array([-1, 0, 1]), order, order)
    return func


# 定义g(x)
def g(x_in):
    return np.exp(x_in) * np.cos([2 * i for i in x_in])


# 读入Laplacian矩阵和状态方程的初值
L = laplace_matrix(M)
y0 = g(x)

# 求解Ricatti方程
# 创建K(t)
K = np.zeros((n_iters + 1, M, M))

# 创建稀疏矩阵speye
row = np.arange(M)
col = np.arange(M)
data = np.ones(M)
speye = np.array(coo_matrix((data, (row, col))))
for n in range(n_iters, 0, -1):
    rhs = -1 / h * np.dot(K[n, :, :], K[n, :, :]) + 2 * K[n, :, :] * L + h * speye
    K[n - 1, :, :] = K[n, :, :] + rhs * dt

print('---------------------------------------------------------------------------')
print('Ricatti方程求解完毕！')
print('---------------------------------------------------------------------------')

# 求解状态方程
# 创建y
y = np.zeros((M, n_iters + 1))
# 读入状态方程的初值
y[:, 0] = y0

for n in range(n_iters):
    y[:, n + 1] = y[:, n] + (L * y[:, n] - 1 / h * np.dot(K[n, :, :], y[:, n])) * dt

print('---------------------------------------------------------------------------')
print('状态方程求解完毕！')
print('---------------------------------------------------------------------------')

# 计算控制变量和目标泛函值
# 创建u
u = np.zeros((M, n_iters + 1))

for n in range(n_iters + 1):
    u[:, n] = -1 / h * np.dot(K[n, :, :], y[:, n])

obj_val = 0.5 * np.dot(np.dot(y[:, 0], K[0, :, :]), y[:, 0].T)

print('---------------------------------------------------------------------------')
print('目标泛函 J = ', obj_val)
print('---------------------------------------------------------------------------')

# 作图 状态变量y
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x, t = np.meshgrid(x, t)
ax.plot_surface(x, t, y.T, cmap='rainbow')
plt.show()

# 作图 控制变量u
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# x, t = np.meshgrid(x, t)
# ax.plot_surface(x, t, u.T, cmap='rainbow')
# plt.show()
