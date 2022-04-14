# 利用变步长梯度下降算法计算扩散方程最优控制问题
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

u = np.zeros((M, n_iters + 1))


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


# 向前欧拉方法求解状态方程, 用以计算最优控制问题
def solve_states(y0, u, L, n_iters):
    global dt
    y = np.zeros((L.shape[0], n_iters + 1))
    # 读入状态方程的初值
    y[:, 0] = y0
    for n in range(0, n_iters):
        y[:, n + 1] = y[:, n] + (L * y[:, n] + u[:, n]) * dt
    return y


# 向后欧拉方法求解伴随方程, 用以计算最优控制问题
def solve_adjoints(y, L, n_iters):
    global h, dt
    p = np.zeros((L.shape[0], n_iters + 1))
    for n in range(n_iters, 0, -1):
        p[:, n - 1] = p[:, n] + (L * p[:, n] + h * y[:, n]) * dt
    return p


# 计算目标泛函值
def object_function(y, u):
    global h, dt
    obj = 0.5 * h * np.sum(np.sum(y ** 2 + u ** 2, axis=1)) * dt
    return obj


# 计算目标泛函的梯度并返回其模
def object_gradient(u, p):
    global h, dt
    obj_grad = h * u + p
    # 针对每一个节点i, 计算目标泛函关于u(t)梯度的范数: 即它们在时间区间上的L2范数, 数值积分使用复化左矩形公式.
    obj_grad_norm = 0.5 * h * np.sum(np.sum(obj_grad[:, 0: obj_grad.shape[1] - 1] ** 2, axis=1)) * dt
    return [obj_grad, obj_grad_norm]


# 优化算法初始步
main_loop = 0
max_loop = 999
obj_err_tol = 1e-9
obj_err = float('inf')
obj_grad_norm_tol = 1e-12

obj_vals = float('inf') * np.ones((max_loop, 1))
obj_grad_norms = float('inf') * np.ones((max_loop, 1))

y = solve_states(y0, u, L, n_iters)
p = solve_adjoints(y, L, n_iters)
obj_vals[main_loop] = object_function(y, u)
[obj_grad, obj_grad_norms[main_loop]] = object_gradient(u, p)

print('-' * 100)
print('mainLoop = ', main_loop, ', J = ', obj_vals[main_loop], ', obj_grad_norm = ', obj_grad_norms[main_loop])

# 优化算法主循环
theta = 1
thetaTol = 1.0e-5
while obj_err > obj_err_tol and main_loop < max_loop:
    main_loop = main_loop + 1

    old_obj_grad = obj_grad
    oldy = y
    oldu = u

    u = u - theta * obj_grad
    y = solve_states(y0, u, L, n_iters)
    p = solve_adjoints(y, L, n_iters)
    obj_vals[main_loop] = object_function(y, u)
    [obj_grad, obj_grad_norms[main_loop]] = object_gradient(u, p)

    obj_err = abs(obj_vals[main_loop] - obj_vals[main_loop - 1])
    print('-' * 100)
    print('mainLoop = ', main_loop, ', J = ', obj_vals[main_loop], ', obj_grad_norm = ', obj_grad_norms[main_loop])
    if obj_grad_norms[main_loop] < obj_grad_norm_tol or theta < thetaTol:
        break

    if obj_vals[main_loop] > obj_vals[main_loop - 1]:
        main_loop = main_loop - 1
        theta = 0.5 * theta
        obj_grad = old_obj_grad
        y = oldy
        u = oldu

print('-' * 100)

# 作图 状态变量y
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x, t = np.meshgrid(x, t)
ax.plot_surface(x, t, y.T, cmap='rainbow')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('状态变量y')
plt.savefig('/Users/mark/Downloads/状态变量y梯度下降.png', dpi=500, bbox_inches='tight')
plt.show()

# 作图 控制变量u
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# x, t = np.meshgrid(x, t)
# ax.plot_surface(x, t, u.T, cmap='rainbow')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('控制变量u')
# plt.savefig('/Users/mark/Downloads/控制变量u梯度下降.png', dpi=500, bbox_inches='tight')
# plt.show()
