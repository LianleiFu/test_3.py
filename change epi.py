from LAPM.linear_autonomous_pool_model import LinearAutonomousPoolModel
from scipy.linalg import expm
from scipy import interpolate
from sympy import Matrix, symbols, Symbol, Function, latex
import os
from numpy import *
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=False)
# from scipy.interpolate import interp1d
# load the compartmental model packages
# from CompartmentalSystems.smooth_reservoir_model import SmoothReservoirModel
# from CompartmentalSystems.pwc_model_run import PWCModelRunfrom
# from IPython.display import display, HTML, Markdown
# from CompartmentalSystems import smooth_reservoir_model
# from CompartmentalSystems import smooth_model_run
# import CompartmentalSystems

########
# model A: inverse Michaelis-Menten kinetics
########
# initial size of pools
k = 0.1
F_npp = 200
K_b = 100
u_b = 0.8
u_l = 0.84
u_s = 0.028
sigma = 4.5
t = arange(0, 650, 1)
X_at = [1715*e**(0.0305*t)/(1715+e**(0.0305*t)-1)+285 for t in t]
Ts = [15 + (sigma/log(2))*log(X_at/285)
      for X_at in X_at]  # global annual mean temperature
list_epi = []
for T in Ts:
    o = 0.39 - 0.016 * (T - 15)
    list_epi.append(o)
# equilibrium carbon in different parameters
C_l_equ_A = [((1-k)*F_npp)/u_l + ((o**(-1)-1) *
                                  (1-k)*u_b*K_b)/u_l for o in list_epi]
C_b_equ_A = [F_npp/((o**(-1)-1)*u_b) for o in list_epi]
C_s_equ_A = [(k+1/(o**(-1)-1))*F_npp/u_s + (1+k*(o**(-1)-1))
             * u_b*K_b/u_s for o in list_epi]

# initial condition
C_l_ini_A = ((1-k)*F_npp)/u_l + ((list_epi[0]**(-1)-1)*(1-k)*u_b*K_b)/u_l
C_b_ini_A = F_npp/((list_epi[0]**(-1)-1)*u_b)
C_s_ini_A = (k+1/(list_epi[0]**(-1)-1))*F_npp/u_s + \
    (1+k*(list_epi[0]**(-1)-1))*u_b*K_b/u_s
list_C_l_A = []
list_C_s_A = []
list_C_b_A = []
for o in list_epi:
    del_C_l_A = -u_l * C_b_ini_A / \
        (C_b_ini_A + K_b) * C_l_ini_A + (1-k) * F_npp
    del_C_s_A = -u_s * C_b_ini_A / \
        (C_b_ini_A + K_b) * C_s_ini_A + u_b * C_b_ini_A + 0.1 * F_npp
    del_C_b_A = o * u_l * C_b_ini_A / (C_b_ini_A + K_b) * C_l_ini_A + \
        o * u_s * C_b_ini_A / (C_b_ini_A + K_b) * C_s_ini_A - u_b * C_b_ini_A
    C_l_ini_A = C_l_ini_A + del_C_l_A
    C_s_ini_A = C_s_ini_A + del_C_s_A
    C_b_ini_A = C_b_ini_A + del_C_b_A
    list_C_l_A.append(C_l_ini_A)
    list_C_s_A.append(C_s_ini_A)
    list_C_b_A.append(C_b_ini_A)

# plot
plt.figure(figsize=(10, 7))
plt.title('Inverse Michaelis-Menten kinetics')
plt.plot(arange(2000, 2650), list_C_l_A, color='purple', label='litter carbon')
plt.plot(arange(2000, 2650), list_C_s_A, color='green', label='soil carbon')
plt.plot(arange(2000, 2650), list_C_b_A, color='blue',
         label='microbial biomass carbon')
plt.xlim([2000, 2650])
plt.ylim([0, 15000])
plt.legend(loc=2)
plt.xlabel('Time (yr)')
plt.ylabel('Mass (g C m -2)')
plt.savefig('E:\\论文\\模型\\Wang\\Wang YP A epi.png')
plt.show()

########
# model B: regular Michaelis-Menten kinetics
########
# initial size of pools
V_l = 172
V_s = 32
K_l = 67275
K_s = 363871

C_l_equ_B = [K_l/(o*V_l/((1-o)*(1-k)*u_b)-1) for o in list_epi]
C_b_equ_B = [F_npp/(u_b*(o**(-1)-1)) for o in list_epi]
C_s_equ_B = [K_s/(V_s/u_b*o/(o+0.1*(1-o))-1) for o in list_epi]

# initial condition
C_l_ini_B = K_l/(list_epi[0]*V_l/((1-list_epi[0])*(1-k)*u_b)-1)
C_b_ini_B = F_npp/(u_b*(list_epi[0]**(-1)-1))
C_s_ini_B = K_s/(V_s/u_b*list_epi[0]/(list_epi[0]+k*(1-list_epi[0]))-1)

list_C_l_B = []
list_C_s_B = []
list_C_b_B = []
for o in list_epi:
    del_C_l_B = (1-k) * F_npp - C_b_ini_B * V_l * C_l_ini_B/(C_l_ini_B + K_l)
    del_C_s_B = k * F_npp + u_b * C_b_ini_B - \
        C_b_ini_B * V_s * C_s_ini_B/(C_s_ini_B + K_s)
    del_C_b_B = o * C_b_ini_B * \
        (V_l*C_l_ini_B/(C_l_ini_B+K_l)+V_s *
         C_s_ini_B/(C_s_ini_B+K_s)) - u_b * C_b_ini_B
    C_l_ini_B = C_l_ini_B + del_C_l_B
    C_s_ini_B = C_s_ini_B + del_C_s_B
    C_b_ini_B = C_b_ini_B + del_C_b_B
    list_C_l_B.append(C_l_ini_B)
    list_C_s_B.append(C_s_ini_B)
    list_C_b_B.append(C_b_ini_B)

plt.figure(figsize=(10, 7))
plt.title('Regular Michaelis-Menten kinetics')
plt.plot(arange(2000, 2650), list_C_l_B, color='purple', label='litter carbon')
plt.plot(arange(2000, 2650), list_C_s_B, color='green', label='soil carbon')
plt.plot(arange(2000, 2650), list_C_b_B, color='blue',
         label='microbial biomass carbon')
plt.xlim([2000, 2650])
plt.ylim([0, 15000])
plt.legend(loc=2)
plt.xlabel('Time (yr)')
plt.ylabel('Mass (g C m -2)')
plt.savefig('E:\\论文\\模型\\Wang\\Wang YP B epi.png')
plt.show()


#########
# plot 3D age density figure for model A
#########
C_l_equ_A = ((1-k)*F_npp)/u_l + ((list_epi[0]**(-1)-1)*(1-k)*u_b*K_b)/u_l
C_b_equ_A = F_npp/((list_epi[0]**(-1)-1)*u_b)
C_s_equ_A = (k+1/(list_epi[0]**(-1)-1))*F_npp/u_s + \
    (1+k*(list_epi[0]**(-1)-1))*u_b*K_b/u_s

B0_A = mat([
    [-u_l*C_b_equ_A/(C_b_equ_A+K_b), 0, 0],
    [0, -u_s*C_b_equ_A/(C_b_equ_A+K_b), u_b],
    [list_epi[0]*u_l*C_b_equ_A /
     (C_b_equ_A+K_b), list_epi[0]*u_s*C_b_equ_A/(C_b_equ_A+K_b), -u_b]
])
B_A = [mat([[-u_l*C_b/(C_b+K_b), 0, 0],
            [0, -u_s*C_b/(C_b+K_b), u_b],
            [epi*u_l*C_b/(C_b+K_b), epi*u_s*C_b/(C_b*K_b), -u_b]
            ]) for C_b, epi in zip(list_C_b_A, list_epi)]
U_t = mat([
    [(1-k)*F_npp],
    [k*F_npp],
    [0]
])
x0 = -B0_A.I*U_t
X0_A = mat([[C_l_equ_A, 0, 0],
            [0, C_s_equ_A, 0],
            [0, 0, C_b_equ_A]])
t0 = 2000
fig = plt.figure()
ax = Axes3D(fig)
t = range(2000, 2650, 1)
a = range(0, 251, 1)
T, A = meshgrid(t, a)  # 网格的创建，这个是关键
T = ravel(T)  # 向量化矩阵
A = ravel(A)
# 计算Z轴，age density
list_p_at_A = []
for a, t in zip(A, T):
    if a >= t - t0:
        p_at_A = expm((t - 2000) * B_A[t - t0]) * X0_A * \
            X0_A.I * expm((a - t + 2000) * B0_A) * U_t
    else:
        p_at_A = expm(a * B_A[t - t0]) * U_t
    list_p_at_A.append(p_at_A)
P_at_A = hstack(list_p_at_A)  # 合并矩阵
T = T.reshape(251, 650)
A = A.reshape(251, 650)
Z = P_at_A.T[:, 1]
Z = Z.reshape(251, 650).A  # 转换成251乘650的array
plt.xlabel('Time(yr)')
plt.ylabel('Age(yr)')
plt.title("Only change epi in model A")
ax.plot_surface(T, A, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# ax.plot(range(2000,2650,1), [80.06865 for a in a], zs=0, zdir='z')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.scatter(T, A, Z, c='c', marker='.')
ax.view_init(elev=20., azim=135)  # 调整视角，elev高程，azim方位角
# plt.savefig('F:\\论文\\模型\\3D plot A.png')
plt.show()


#########
# plot 3D age density figure for model B
#########
C_l_equ_B = K_l/(list_epi[0]*V_l/((1-list_epi[0])*(1-k)*u_b)-1)
C_b_equ_B = F_npp/(u_b*(list_epi[0]**(-1)-1))
C_s_equ_B = K_s/(V_s/u_b*list_epi[0]/(list_epi[0]+k*(1-list_epi[0]))-1)

B0_B = mat([
    [-C_b_equ_B*V_l/(C_l_equ_B+K_l), 0, 0],
    [0, -C_b_equ_B*V_s/(C_s_equ_B+K_s), u_b],
    [list_epi[0]*C_b_equ_B*V_l /
     (C_l_equ_B+K_l), list_epi[0]*C_b_equ_B*V_s/(C_s_equ_B+K_s), -u_b]
])
B_B = [mat([[-C_b*V_l/(C_l+K_l), 0, 0],
            [0, -C_b*V_s/(C_s+K_s), u_b],
            [epi*C_b*V_l/(C_l+K_l), epi*C_b*V_s/(C_s+K_s), -u_b]
            ]) for C_b, C_l, C_s, epi in zip(list_C_b_B, list_C_l_B, list_C_s_B, list_epi)]
U_t = mat([
    [(1-k)*F_npp],
    [k*F_npp],
    [0]
])
x0 = -B0_B.I*U_t
X0_B = mat([[C_l_equ_B, 0, 0],
            [0, C_s_equ_B, 0],
            [0, 0, C_b_equ_B]])
t0 = 2000
fig = plt.figure()
ax = Axes3D(fig)
t = range(2000, 2650, 1)
a = range(0, 251, 1)
T, A = meshgrid(t, a)  # 网格的创建，这个是关键
T = ravel(T)  # 向量化矩阵
A = ravel(A)
# 计算Z轴，age density
list_p_at_B = []
for a, t in zip(A, T):
    if a >= t - t0:
        p_at_B = expm((t - 2000) * B_B[t - t0]) * X0_B * \
            X0_B.I * expm((a - t + 2000) * B0_B) * U_t
    else:
        p_at_B = expm(a * B_B[t - t0]) * U_t
    list_p_at_B.append(p_at_B)
P_at_B = hstack(list_p_at_B)  # 合并矩阵
T = T.reshape(251, 650)
A = A.reshape(251, 650)
Z = P_at_B.T[:, 1]
Z = Z.reshape(251, 650).A  # 转换成251乘736的array
plt.xlabel('Time(yr)')
plt.ylabel('Age(yr)')
plt.title("Only change epi in model B")
ax.plot_surface(T, A, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# ax.plot(range(2000,2650,1), [80.06865 for a in a], zs=0, zdir='z')
# ax.scatter(T, A, Z, c='c', marker=',')
ax.view_init(elev=20., azim=135)  # 调整视角，elev高程，azim方位角
# plt.savefig('F:\\论文\\模型\\3D plot B.png')
plt.show()


#########
# Equilibrium Age Densities model A: the last year
#########
a = arange(0, 251, 1)
C_l_equ_A_649st = ((1-k)*F_npp)/u_l + \
    ((list_epi[649]**(-1)-1)*(1-k)*u_b*K_b)/u_l
C_b_equ_A_649st = F_npp/((list_epi[649]**(-1)-1)*u_b)
C_s_equ_A_649st = (k+1/(list_epi[649]**(-1)-1)) * \
    F_npp/u_s + (1+k*(list_epi[649]**(-1)-1))*u_b*K_b/u_s

B0_A_649st = mat([
    [-u_l*C_b_equ_A_649st/(C_b_equ_A_649st+K_b), 0, 0],
    [0, -u_s*C_b_equ_A_649st/(C_b_equ_A_649st+K_b), u_b],
    [list_epi[649]*u_l*C_b_equ_A_649st /
     (C_b_equ_A_649st+K_b), list_epi[649]*u_s*C_b_equ_A_649st/(C_b_equ_A_649st+K_b), -u_b]
])
u0_A = mat([
           [(1-k)*F_npp],
           [k*F_npp],
           [0]
           ])
# x0 = -B0**(-1)*u0
X0_A_649st = mat([[C_l_equ_A_649st, 0, 0],
                  [0, C_s_equ_A_649st, 0],
                  [0, 0, C_b_equ_A_649st]])
p0_a_A_649st = [X0_A_649st.I*expm(i*B0_A_649st)*u0_A for i in a]
P0_a_A_649st = hstack(p0_a_A_649st)
plt.figure(figsize=(10, 7))
plt.title('Only change epi in model A')
plt.plot(a, C_l_equ_A_649st *
         P0_a_A_649st.T[:, 0], color='blue', label='litter carbon')
plt.plot(a, C_s_equ_A_649st *
         P0_a_A_649st.T[:, 1], color='green', label='soil carbon')
plt.plot(a, C_b_equ_A_649st *
         P0_a_A_649st.T[:, 2], color='purple', label='microbial biomass carbon')
plt.plot(a, C_l_equ_A_649st*P0_a_A_649st.T[:, 0] + C_s_equ_A_649st*P0_a_A_649st.T[:,
        1] + C_b_equ_A_649st*P0_a_A_649st.T[:, 2], color='red', label='Total')
plt.legend(loc=1)
plt.xlim([0, 250])
plt.ylim([0, 250])
plt.xlabel('Age (yr)')
plt.ylabel('Mass (PgC/yr)')
plt.savefig('E:\\论文\\模型\\Wang\\the last year\\epi model A.png')
plt.show()


#########
# Equilibrium Age Densities model B: the last year
#########
C_l_equ_B_649st = K_l/(list_epi[649]*V_l/((1-list_epi[649])*(1-k)*u_b)-1)
C_b_equ_B_649st = F_npp/(u_b*(list_epi[649]**(-1)-1))
C_s_equ_B_649st = K_s / \
    (V_s/u_b*list_epi[649]/(list_epi[649]+k*(1-list_epi[649]))-1)

B0_B_649st = mat([
    [-C_b_equ_B_649st*V_l/(C_l_equ_B_649st+K_l), 0, 0],
    [0, -C_b_equ_B_649st*V_s/(C_s_equ_B_649st+K_s), u_b],
    [list_epi[649]*C_b_equ_B_649st*V_l /
     (C_l_equ_B_649st+K_l), list_epi[649]*C_b_equ_B_649st*V_s/(C_s_equ_B_649st+K_s), -u_b]
])
u0_B = mat([
           [(1-k)*F_npp],
           [k*F_npp],
           [0]
           ])
# x0 = -B0**(-1)*u0
X0_B_649st = mat([[C_l_equ_B_649st, 0, 0],
                  [0, C_s_equ_B_649st, 0],
                  [0, 0, C_b_equ_B_649st]])
p0_a_B_649st = [X0_B_649st.I*expm(i*B0_B_649st)*u0_B for i in a]
P0_a_B_649st = hstack(p0_a_B_649st)
plt.figure(figsize=(10, 7))
plt.title('Only change epi in model B')
plt.plot(a, C_l_equ_B_649st *
         P0_a_B_649st.T[:, 0], color='blue', label='litter carbon')
plt.plot(a, C_s_equ_B_649st *
         P0_a_B_649st.T[:, 1], color='green', label='soil carbon')
plt.plot(a, C_b_equ_B_649st *
         P0_a_B_649st.T[:, 2], color='purple', label='microbial biomass carbon')
plt.plot(a, C_l_equ_B_649st*P0_a_B_649st.T[:, 0] + C_s_equ_B_649st*P0_a_B_649st.T[:,
        1] + C_b_equ_B_649st*P0_a_B_649st.T[:, 2], color='red', label='Total')
plt.legend(loc=1)
plt.xlim([0, 250])
plt.ylim([0, 250])
plt.xlabel('Age (yr)')
plt.ylabel('Mass (PgC/yr)')
plt.savefig('E:\\论文\\模型\\Wang\\the last year\\epi model B.png')
plt.show()
