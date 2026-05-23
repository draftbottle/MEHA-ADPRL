# 多电液伺服系统反馈线性化无模型反步最优一致性控制算法仿真实现(已实现仿真)
# 1.电液伺服系统反馈线性化模型
# 2.Backstepping Method and Reinforcement Learning,可调节神经网络模型
# 3.实现了机械臂恒定值阶跃末端轨迹和正弦波末端轨迹
# 4.使用plotly绘图,网页交互展示结果图,仿照ieee trans出图
# 5.钢轨打磨精度要求0.005mm-0.01mm
# 6.第二步反步控制已加入强化学习设计
# 7.已经实现系统模型未知动态NN拟合
# 8.TODO:未来将实现外部强干扰力NN估计,扰动估计部分代码需要继续修改(is Done)
# 9.note:已经过仿真验证该算法从理论上是有效,但效果不太好,需要后期继续调试
# 10.NOTE:新版本剔除了事件触发机制,该机制对不确定性模型不友好,无法做到大幅度减少通信次数
# 11.已完成外部扰动观测,实现了无模型EHA位置跟踪,但是精度不够,需后期继续优化代码和理论
# 12.TODO:下一步计划剔除反馈线性化,直接基于原始状态方程实现该算法！！！！！！@@@@@@########
# 13.NOTE:使用matplotlib绘图,适合论文写作
# 14.NOTE:修复扰动估计值无法正常绘制的问题
# 15.TODO:需要加入性能函数转换第一步误差实现预设性能控制

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize, sparse, special
from scipy.integrate import solve_ivp
import os

# 创建保存图片的目录
os.makedirs("EHA_fig", exist_ok=True)

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class EHAAgent:
    """电液伺服系统智能体类（基于反馈线性化）"""

    def __init__(self, agent_id, initial_state, gamma1, gamma2, gamma3, centers, q, rho, kappa_c, kappa_a, kappa_d, sigma):
        """
        初始化EHA智能体
        :param agent_id: 智能体ID
        :param initial_state: 初始状态 [z1, z2, z3] = [位移, 速度, 压力]
        :param gamma1, gamma2, gamma3: 反步控制增益
        :param centers: RBF中心点
        :param rho: RBF宽度
        :param kappa_c: Critic学习率
        :param kappa_a: Actor学习率
        :param sigma: 正则化参数
        """
        self.id = agent_id
        self.state = np.array(initial_state)  # [z1, z2, z3] = [位移, 速度, 压力]
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3

        # EHA系统参数（来自论文表1）
        self.A = 1.83e-3  # 液压缸活塞有效作用面积(m^2)
        self.m = 200  # 等效负载质量(kg)
        self.K = 2e3  # 等效负载刚度(N/m)
        self.B_val = 3e4  # 等效负载阻尼(N·s/m)-重命名避免与矩阵B冲突
        self.F = 1e3  # 外负载力(N)
        self.Ps = 10e6  # 供油压力(Pa)
        self.Ci = 2e-11  # 液压缸总泄露系数(m³·s⁻¹·Pa⁻¹)
        self.V = 1.2e-4  # 液压缸两腔总容积(m³)
        self.beta_e = 6.86e8  # 液压油体积弹性模量(N/m²)
        self.Cd = 0.6  # 伺服阀阀口流量系数
        self.omega = 3e-2  # 伺服阀阀口面积梯度(m)
        self.rho_oil = 900  # 液压油密度(kg/m³)-重命名避免与RBF的rho冲突
        self.Kv = 3e-4  # 增益系数(m/V)
        self.F_L = 150  # EHA外扰动力(150N), todo:需调试NN观测器辨识扰动

        # RBF神经网络参数
        self.centers = centers
        self.rho = rho
        self.q = q

        # 强化学习参数
        self.kappa_c = kappa_c
        self.kappa_a = kappa_a
        self.kappa_d = kappa_d
        self.sigma = sigma

        # 神经网络权重初始化
        self.Mc = 0.9 * np.ones(self.q)  # Critic权重
        self.Ma = 0.4 * np.ones(self.q)  # Actor权重
        self.Mf = 0.5 * np.ones(self.q)  # 未知动力学
        self.Md = 0.8 * np.ones(self.q)  # 未知外界扰动

        # ========== 新增：第二步的神经网络权重 ==========
        self.Mc2 = 0.9 * np.ones(self.q)  # 第二步Critic权重
        self.Ma2 = 0.4 * np.ones(self.q)  # 第二步Actor权重
        self.Mf2 = 0.5 * np.ones(self.q)  # 第二步未知动力学
        self.Md2 = 0.8 * np.ones(self.q)  # 第二步未知外界扰动

        # 控制变量
        self.alpha1 = 0.0
        self.alpha2 = 0.0
        self.u_linear = 0.0  # 线性化空间的控制量
        self.v_actual = 0.0  # 实际控制电压
        self.xi1 = 0.0
        self.xi2 = 0.0
        self.xi3 = 0.0

        # 新增：扰动观测值存储
        self.d2_hat = 0.0  # 第二步扰动观测值
        self.d_hat = 0.0  # 第三步扰动观测值

    def rbf_basis(self, x):
        """计算RBF基函数"""
        return np.exp(-(x - self.centers) ** 2 / (2 * self.rho ** 2))

    def compute_feedback_linearization_terms(self):
        """计算反馈线性化相关项"""
        z1, z2, z3 = self.state

        # 计算李导数项
        Lf3h = (self.B_val * self.K / self.m ** 2) * z1 + \
               (z2 / self.m) * (self.B_val ** 2 / self.m - self.K - (4 * self.A ** 2 * self.beta_e) / self.V) - \
               (z3 / self.m) * ((self.A * self.B_val / self.m) + (4 * self.A * self.beta_e * self.Ci) / self.V) + \
               (self.B_val * self.F) / self.m ** 2

        # 计算LgLf²h(z)项
        sign_v = np.sign(self.v_actual)
        LgLf2h = (4 * self.A * self.beta_e * self.Cd * self.omega * self.Kv) / (self.m * self.V) * np.sqrt(np.abs(self.Ps - sign_v * z3) / self.rho_oil)

        return Lf3h, LgLf2h

    def virtual_control_step1(self, neighbor_states, xr, dxr, B_ii):
        """
        第一步反步：计算虚拟控制α1
        :param neighbor_states: 邻居智能体的z1状态（位移）
        :param xr: 参考信号
        :param dxr: 参考信号导数
        :param B_ii: 与领导者的通信权重
        """
        z1 = self.state[0]

        # 计算一致性误差
        consensus_term = np.sum(z1 - np.array(neighbor_states))
        self.xi1 = B_ii * (z1 - xr) + consensus_term
        self.alpha1 = -self.gamma1 * self.xi1 + dxr  # 修正：加入参考信号导数

        # 计算α1的导数近似
        #dalpha1_dt = -self.gamma1 * (-dxr * B_ii)
        #return dalpha1_dt

    def rl_virtual_control_step2(self):
        """
        第二步反步：基于强化学习的虚拟控制α2
        :param dalpha1_dt: α1的导数
        :return: 第二步Critic和Actor的权重导数
        """
        z2 = self.state[1]
        self.xi2 = z2 - self.alpha1
        # print(self.xi2)

        # RBF基函数
        phi2 = self.rbf_basis(self.xi2)
        phi_f2 = self.rbf_basis(self.xi2)

        # f未知动力学
        kk2 = 50  # kk2 = k2
        f2 = self.Mf2 @ phi_f2
        d2 = (self.gamma2 * self.xi2 + 0.5 * self.Md2 @ phi2 + f2) * kk2    # NOTE:扰动观测器设计 +0.75
        self.d2_hat = d2

        # Actor控制输出(第二步的虚拟控制量α2)
        self.alpha2 = -self.gamma2 * self.xi2 - 0.5 * self.Ma2 @ phi2 - f2 - d2

        # 第二步Critic更新
        phi2_phi2_T = np.outer(phi2, phi2)
        regularization = self.sigma * np.eye(self.q)
        dMc2 = -self.kappa_c * (phi2_phi2_T + regularization) @ self.Mc2

        # 第二步Actor更新
        term = self.kappa_a * (self.Ma2 - self.Mc2) + self.kappa_c * self.Mc2
        dMa2 = -(phi2_phi2_T + regularization) @ term

        # 第二步Disturbance NN权重更新
        term = self.kappa_d * (self.Md2 - self.Mc2) - self.kappa_c * self.Md2
        dMd2 = (phi2_phi2_T + regularization) @ term

        # 第二步f权重更新
        k2 = 50  # k2 = kk2
        delta_i2 = 25   # 16
        dMf2 = (1 - k2) * phi_f2 * self.xi2 - delta_i2 * self.Mf2

        return dMc2, dMa2, dMf2, dMd2

    def rl_optimized_control(self, current_time):
        """
        第三步：基于强化学习的优化控制(在线性化空间)
        current_time: 仿真时间变量
        return: 状态导数, Critic权重导数, Actor权重导数
        """
        z1, z2, z3 = self.state

        # 计算线性化空间的状态x3
        x3 = (self.A * z3 - self.B_val * z2 - self.K * z1 - self.F) / self.m
        self.xi3 = x3 - self.alpha2
        # print(self.xi3)

        # RBF基函数
        phi = self.rbf_basis(self.xi3)
        phi_f = self.rbf_basis(self.xi3)

        # f未知动力学
        kk = 50
        f = self.Mf @ phi_f
        d = (self.gamma3 * self.xi3 + 0.5 * self.Md @ phi + f) * kk    # note:hat_d扰动观测器
        self.d_hat = d

        # Actor控制输出（线性化空间的控制量u）
        self.u_linear = -self.gamma3 * self.xi3 - 0.5 * self.Ma @ phi - f - d

        # 计算反馈线性化项
        Lf3h, LgLf2h = self.compute_feedback_linearization_terms()

        # 计算实际控制电压v
        if abs(LgLf2h) > 1e-11:  # 避免除零
            self.v_actual = (self.u_linear - Lf3h) / LgLf2h
        else:
            self.v_actual = 0.0

        # Critic权重更新
        phi_phi_T = np.outer(phi, phi)
        regularization = self.sigma * np.eye(self.q)
        dMc = -self.kappa_c * (phi_phi_T + regularization) @ self.Mc

        # Actor权重更新
        term = self.kappa_a * (self.Ma - self.Mc) + self.kappa_c * self.Mc
        dMa = -(phi_phi_T + regularization) @ term

        # Disturbance NN权重更新
        term = self.kappa_d * (self.Md - self.Mc) - self.kappa_c * self.Md
        dMd = (phi_phi_T + regularization) @ term

        # 第二步f权重更新
        k = 50  # 此处该值取的是原始公式中参数k的倒数
        delta_i3 = 25
        dMf = (1 - k) * phi_f * self.xi3 - delta_i3 * self.Mf

        # EHA系统状态导数（使用实际控制电压）
        F_L = (400*np.sin(11 * np.pi * current_time) + 200*np.cos(9 * np.pi * current_time) + 100*np.sin(5 * np.pi * current_time)
               + 50*np.cos(7 * current_time) + 20*np.sin(4 * current_time) + 7*np.sin(1.25 * current_time))   # 复杂外扰动
        dz1 = z2
        dz2 = (self.A * z3 - self.B_val * z2 - self.K * z1 - self.F) / self.m - F_L / self.m - 75 * z2 - 5 * z1  # NOTE:加入了系统非线性动态

        delta_Cd = 0.2 # 0.2
        delta_omega = 0.16  # 系统部分不确定参数
        delta_rho_oil = -0.1 # -0.1
        delta_beta_e = 0.4 # 0.35
        delta_Ci = 0.25 # 0.25
        # 计算负载流量Q（使用当前控制电压）
        sign_v = np.sign(self.v_actual)
        Q = (1 + delta_Cd) * self.Cd * (1 + delta_omega) * self.omega * self.Kv * self.v_actual * np.sqrt(np.abs(self.Ps - sign_v * z3) / ((1+delta_rho_oil) * self.rho_oil))
        dz3 = (4 * (1 + delta_beta_e) * self.beta_e / self.V) * (Q - self.A * z2 - (1 + delta_Ci) * self.Ci * z3) # NOTE:加入了系统非线性动态

        return [dz1, dz2, dz3], dMc, dMa, dMf, dMd

    def compute_control_input(self):
        """计算控制输入（不更新状态）"""
        z1, z2, z3 = self.state

        # 计算线性化空间的状态x3
        x3 = (self.A * z3 - self.B_val * z2 - self.K * z1 - self.F) / self.m
        xi3 = x3 - self.alpha2

        # RBF基函数
        phi = self.rbf_basis(xi3)
        phi_f = self.rbf_basis(xi3)

        # f未知动力学
        f = self.Mf @ phi_f
        d = (self.gamma3 * xi3 + 0.5 * self.Md @ phi + f) * 50  # note:hat_d, k=4
        self.d_hat = d  # NOTE:<--- 必须加上这行，analyze_results才能读到数据

        u_linear = -self.gamma3 * xi3 - 0.5 * self.Ma @ phi - f - self.d_hat
        # 计算反馈线性化项
        Lf3h, LgLf2h = self.compute_feedback_linearization_terms()

        # 计算实际控制电压
        if abs(LgLf2h) > 1e-11:
            v_actual = (u_linear - Lf3h) / LgLf2h
        else:
            v_actual = 0.0

        # 条件表达式确保v_actual在min_value和max_value之间
        # v_actual = -10 if v_actual < -10 else (10 if v_actual > 10 else v_actual)
        return v_actual

    def get_linear_state(self):
        """获取线性化空间的状态"""
        z1, z2, z3 = self.state
        x1 = z1
        x2 = z2
        x3 = (self.A * z3 - self.B_val * z2 - self.K * z1 - self.F) / self.m
        return np.array([x1, x2, x3])


class EHAMultiAgentSystem:
    """基于EHA的多智能体系统类"""

    def __init__(self, n=6, m=3, q=40):
        """
        初始化EHA多智能体系统
        :param n: 智能体数量
        :param m: 系统阶数
        :note:每次打磨切削量为0.02毫米，打磨后的粗糙度不大于10微米(即误差不能超过0.01mm)
        """
        self.n = n
        self.m = m
        self.agents = []
        self.t_span = [0, 1]  # 仿真时间5s

        # 通信拓扑矩阵 (环形拓扑)
        self.A = np.array([
            [0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0]
        ])

        # 通信权重矩阵 (智能体2、5与领导者通信)
        self.B = np.diag([0, 1, 0, 1, 0, 1])

        # 初始化智能体（使用物理合理的初始状态）
        initial_states = [
            [0.000025, 0.0, 4e5],  # 小位移，零速度，中等压力
            [0.00002, 0.0, 3e5],
            [0.000025, 0.0, 3.5e5],
            [0.000035, 0.0, 3.25e5],
            [0.00002, 0.0, 4.5e5],
            [0.00004, 0.0, 4.25e5]
        ]

        # 公共参数
        self.q = q  # RBF神经元数量
        centers = np.linspace(-10, 10, q)  # RBF中心
        rho = 0.5010  # RBF宽度

        for i in range(n):
            agent = EHAAgent(
                agent_id=i,
                initial_state=initial_states[i],
                gamma1=410.8678, gamma2=281.5765, gamma3=115.0284,
                centers=centers, q=q, rho=rho,
                kappa_c=67.1180, kappa_a=44.6475, kappa_d=4, sigma=0.8351  # note:kappa_d需重新调试此参数
            )

            self.agents.append(agent)

    def reference_signal(self, t):
        """参考信号0.05mm(适应EHA物理范围),注释信号为电子科大郭庆组实验参考信号,经验证误差在1.5mm附近,和文献误差不大,此代码正确"""
        return 0.00005 # * np.sin(2 * t)
        #return 0.05 * np.sin(np.pi * t) + 0.02 * np.sin(2 * np.pi * t)

    def reference_derivative(self, t):
        """参考信号导数,钢轨打磨机械臂为恒定位置打磨"""
        return 0 #.00005 * 2 * np.cos(2 * t)
        #return 0.05 * np.pi * np.cos(np.pi * t) + 0.02 * np.pi * np.cos(2 * np.pi * t)

    def system_dynamics(self, t, X):
        """EHA系统动力学ODE函数"""
        # 解析状态向量 - 现在包含8组权重
        states = X[:self.n * self.m].reshape(self.n, self.m)
        Mc_all = X[self.n * self.m:self.n * self.m + self.q * self.n].reshape(self.q, self.n)
        Ma_all = X[self.n * self.m + self.q * self.n:self.n * self.m + 2 * self.q * self.n].reshape(self.q, self.n)
        Mc2_all = X[self.n * self.m + 2 * self.q * self.n:self.n * self.m + 3 * self.q * self.n].reshape(self.q, self.n)
        Ma2_all = X[self.n * self.m + 3 * self.q * self.n:self.n * self.m + 4 * self.q * self.n].reshape(self.q, self.n)
        Mf2_all = X[self.n * self.m + 4 * self.q * self.n:self.n * self.m + 5 * self.q * self.n].reshape(self.q, self.n)
        Md2_all = X[self.n * self.m + 5 * self.q * self.n:self.n * self.m + 6 * self.q * self.n].reshape(self.q, self.n)  # 新增：第二步扰动权重
        Mf_all = X[self.n * self.m + 6 * self.q * self.n:self.n * self.m + 7 * self.q * self.n].reshape(self.q, self.n)
        Md_all = X[self.n * self.m + 7 * self.q * self.n:].reshape(self.q, self.n)  # 新增：第三步扰动权重

        # 更新智能体状态
        for i, agent in enumerate(self.agents):
            agent.state = states[i]
            agent.Mc = Mc_all[:, i]
            agent.Ma = Ma_all[:, i]
            agent.Mc2 = Mc2_all[:, i]
            agent.Ma2 = Ma2_all[:, i]
            agent.Mf2 = Mf2_all[:, i]
            agent.Md2 = Md2_all[:, i]  # 新增
            agent.Mf = Mf_all[:, i]
            agent.Md = Md_all[:, i]  # 新增

        # 参考信号
        xr = self.reference_signal(t)
        dxr = self.reference_derivative(t)

        # 导数初始化
        dx = np.zeros((self.n, self.m))
        dMc = np.zeros((self.q, self.n))
        dMa = np.zeros((self.q, self.n))
        dMc2 = np.zeros((self.q, self.n))
        dMa2 = np.zeros((self.q, self.n))
        dMf2 = np.zeros((self.q, self.n))
        dMd2 = np.zeros((self.q, self.n))  # 新增：第二步扰动权重导数
        dMf = np.zeros((self.q, self.n))
        dMd = np.zeros((self.q, self.n))  # 新增：第三步扰动权重导数

        # 对每个智能体执行反步控制
        for i, agent in enumerate(self.agents):
            # 获取邻居智能体的状态
            neighbors = np.where(self.A[i] == 1)[0]
            neighbor_states = [self.agents[j].state[0] for j in neighbors]

            # 第一步：虚拟控制α1
            agent.virtual_control_step1(neighbor_states, xr, dxr, self.B[i, i])

            # 第二步：基于RL的虚拟控制α2
            dMc2_i, dMa2_i, dMf2_i, dMd2_i = agent.rl_virtual_control_step2()  # 修改

            # 保存第二步导数
            dMc2[:, i] = dMc2_i
            dMa2[:, i] = dMa2_i
            dMf2[:, i] = dMf2_i  # 新增
            dMd2[:, i] = dMd2_i  # 新增

            # 第三步：强化学习优化控制
            dx_i, dMc_i, dMa_i, dMf_i, dMd_i = agent.rl_optimized_control(t)  # 修改

            # 保存导数
            dx[i] = dx_i
            dMc[:, i] = dMc_i
            dMa[:, i] = dMa_i
            dMf[:, i] = dMf_i  # 新增
            dMd[:, i] = dMd_i  # 新增

        return np.concatenate([dx.flatten(), dMc.flatten(), dMa.flatten(), dMc2.flatten(), dMa2.flatten(), dMf2.flatten(), dMd2.flatten(), dMf.flatten(), dMd.flatten()])

    def simulate(self):
        """运行仿真"""
        # 初始状态向量 - 现在包含8组权重
        X0 = np.concatenate([
            np.array([agent.state for agent in self.agents]).flatten(),
            0.5 * np.ones(self.q * self.n),  # Mc
            0.4 * np.ones(self.q * self.n),  # Ma
            0.5 * np.ones(self.q * self.n),  # Mc2
            0.4 * np.ones(self.q * self.n),  # Ma2
            0.5 * np.ones(self.q * self.n),  # Mf2
            0.2 * np.ones(self.q * self.n),  # Md2 - 新增：第二步扰动权重
            0.5 * np.ones(self.q * self.n),  # Mf
            0.2 * np.ones(self.q * self.n),  # Md - 新增：第三步扰动权重
        ])

        # 数值积分求解
        print("开始EHA系统数值积分求解...")
        # 使用更严格的求解器设置
        sol = solve_ivp(self.system_dynamics, self.t_span, X0, method='RK45', rtol=1e-6, atol=1e-8, max_step=0.01, dense_output=True)

        print(f"积分完成，时间点数量: {len(sol.t)}")

        return sol

    def analyze_results(self, sol):
        """分析仿真结果"""
        t = sol.t
        X = sol.y.T
        N = len(t)

        # 提取状态 - 修改索引以匹配新的状态向量结构
        states = np.zeros((self.n, self.m, N))
        for k in range(N):
            states[:, :, k] = X[k, :self.n * self.m].reshape(self.n, self.m)

        # 参考信号
        xr = np.array([self.reference_signal(ti) for ti in t])

        # 计算跟踪误差
        tracking_errors = np.zeros((self.n, N))
        for i in range(self.n):
            tracking_errors[i, :] = states[i, 0, :] - xr

        # 控制输入重构
        v_history = np.zeros((self.n, N))
        alpha2_history = np.zeros((self.n, N))
        d2_history = np.zeros((self.n, N))  # 新增：第二步扰动观测值
        d_history = np.zeros((self.n, N))  # 新增：第三步扰动观测值
        for k in range(N):
            # 更新智能体状态 - 修改索引以匹配新的状态向量结构
            states_k = X[k, :self.n * self.m].reshape(self.n, self.m)
            Mc_all = X[k, self.n * self.m:self.n * self.m + self.q * self.n].reshape(self.q, self.n)
            Ma_all = X[k, self.n * self.m + self.q * self.n:self.n * self.m + 2 * self.q * self.n].reshape(self.q, self.n)
            Mc2_all = X[k, self.n * self.m + 2 * self.q * self.n:self.n * self.m + 3 * self.q * self.n].reshape(self.q, self.n)
            Ma2_all = X[k, self.n * self.m + 3 * self.q * self.n:self.n * self.m + 4 * self.q * self.n].reshape(self.q, self.n)
            Mf2_all = X[k, self.n * self.m + 4 * self.q * self.n:self.n * self.m + 5 * self.q * self.n].reshape(self.q, self.n)
            Md2_all = X[k, self.n * self.m + 5 * self.q * self.n:self.n * self.m + 6 * self.q * self.n].reshape(self.q, self.n)  # 新增
            Mf_all = X[k, self.n * self.m + 6 * self.q * self.n:self.n * self.m + 7 * self.q * self.n].reshape(self.q, self.n)
            Md_all = X[k, self.n * self.m + 7 * self.q * self.n:].reshape(self.q, self.n)  # 新增

            for i, agent in enumerate(self.agents):
                agent.state = states_k[i]
                agent.Mc = Mc_all[:, i]
                agent.Ma = Ma_all[:, i]
                agent.Mc2 = Mc2_all[:, i]
                agent.Ma2 = Ma2_all[:, i]
                agent.Mf2 = Mf2_all[:, i]
                agent.Md2 = Md2_all[:, i]  # 新增
                agent.Mf = Mf_all[:, i]
                agent.Md = Md_all[:, i]  # 新增

            # 重新计算控制输入
            for i, agent in enumerate(self.agents):
                # 获取邻居智能体的状态
                neighbors = np.where(self.A[i] == 1)[0]
                neighbor_states = [self.agents[j].state[0] for j in neighbors]

                # 第一步：虚拟控制α1
                agent.virtual_control_step1(neighbor_states, xr[k], self.reference_derivative(t[k]), self.B[i, i])

                # 第二步：基于RL的虚拟控制α2
                agent.rl_virtual_control_step2()
                alpha2_history[i, k] = agent.alpha2
                d2_history[i, k] = agent.d2_hat

                # 第三步：计算控制电压
                v_actual = agent.compute_control_input()
                v_history[i, k] = v_actual
                d_history[i, k] = agent.d_hat

        # 成本函数
        cost_functions = np.zeros((self.n, N))
        for i in range(self.n):
            for k in range(N):
                cost_functions[i, k] = tracking_errors[i, k] ** 2 + v_history[i, k] ** 2 - 50 * d2_history[i, k] ** 2

        # 神经网络权重范数
        Mc_norms = np.zeros((self.n, N))
        Ma_norms = np.zeros((self.n, N))
        Mc2_norms = np.zeros((self.n, N))
        Ma2_norms = np.zeros((self.n, N))
        Mf2_norms = np.zeros((self.n, N))
        Md2_norms = np.zeros((self.n, N))  # 新增：第二步扰动权重范数
        Mf_norms = np.zeros((self.n, N))
        Md_norms = np.zeros((self.n, N))  # 新增：第三步扰动权重范数

        for k in range(N):
            # 修改索引以匹配新的状态向量结构
            Mc_all = X[k, self.n * self.m:self.n * self.m + self.q * self.n].reshape(self.q, self.n)
            Ma_all = X[k, self.n * self.m + self.q * self.n:self.n * self.m + 2 * self.q * self.n].reshape(self.q, self.n)
            Mc2_all = X[k, self.n * self.m + 2 * self.q * self.n:self.n * self.m + 3 * self.q * self.n].reshape(self.q, self.n)
            Ma2_all = X[k, self.n * self.m + 3 * self.q * self.n:self.n * self.m + 4 * self.q * self.n].reshape(self.q, self.n)
            Mf2_all = X[k, self.n * self.m + 4 * self.q * self.n:self.n * self.m + 5 * self.q * self.n].reshape(self.q, self.n)
            Md2_all = X[k, self.n * self.m + 5 * self.q * self.n:self.n * self.m + 6 * self.q * self.n].reshape(self.q, self.n)  # 新增
            Mf_all = X[k, self.n * self.m + 6 * self.q * self.n:self.n * self.m + 7 * self.q * self.n].reshape(self.q, self.n)
            Md_all = X[k, self.n * self.m + 7 * self.q * self.n:].reshape(self.q, self.n)  # 新增

            for i in range(self.n):
                Mc_norms[i, k] = np.linalg.norm(Mc_all[:, i])
                Ma_norms[i, k] = np.linalg.norm(Ma_all[:, i])
                Mc2_norms[i, k] = np.linalg.norm(Mc2_all[:, i])
                Ma2_norms[i, k] = np.linalg.norm(Ma2_all[:, i])
                Mf2_norms[i, k] = np.linalg.norm(Mf2_all[:, i])
                Md2_norms[i, k] = np.linalg.norm(Md2_all[:, i])  # 新增
                Mf_norms[i, k] = np.linalg.norm(Mf_all[:, i])
                Md_norms[i, k] = np.linalg.norm(Md_all[:, i])  # 新增

        return {
            'time': t,
            'states': states,
            'reference': xr,
            'tracking_errors': tracking_errors,
            'control_inputs': v_history,
            'virtual_controls': alpha2_history,
            'cost_functions': cost_functions,
            'critic_norms': Mc_norms,
            'actor_norms': Ma_norms,
            'critic2_norms': Mc2_norms,
            'actor2_norms': Ma2_norms,
            'f2_norms': Mf2_norms,
            'd2_norms': Md2_norms,  # 新增
            'f_norms': Mf_norms,
            'd_norms': Md_norms,  # 新增
            'disturbance_estimates_step2': d2_history,  # 新增：第二步扰动观测值
            'disturbance_estimates_step3': d_history,  # 新增：第三步扰动观测值
        }

    def visualize_results(self, results):
        """通过matplotlib可视化仿真结果，每个图表在单独窗口中显示"""
        t = results['time']
        states = results['states']
        xr = results['reference']
        tracking_errors = results['tracking_errors']
        v_history = results['control_inputs']
        cost_functions = results['cost_functions']
        Mc_norms = results['critic_norms']
        Ma_norms = results['actor_norms']
        Mc2_norms = results['critic2_norms']
        Ma2_norms = results['actor2_norms']
        Md2_norms = results['d2_norms']
        Md_norms = results['d_norms']
        d2_history = results['disturbance_estimates_step2']
        d_history = results['disturbance_estimates_step3']

        print("绘制EHA系统仿真结果...")

        # IEEE Transactions标准配色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11

        # ========== 图表1: 位移跟踪性能 ==========
        print("绘制图1: 3D位移跟踪性能...")
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111, projection='3d')

        # 添加参考平面
        T, A = np.meshgrid(np.linspace(min(t), max(t), 50), np.linspace(-0.5, self.n - 0.5, self.n + 2))
        Z = np.full_like(T, self.reference_signal(0) * 1000)

        # 绘制参考平面
        surf = ax1.plot_surface(T, A, Z, alpha=0.3, color='green', label='Reference Plane')

        # 绘制参考信号线
        ax1.plot(t, [-0.3] * len(t), xr * 1000, linewidth=3, color='darkgreen', linestyle='--', label='Reference Signal')

        # 绘制每个智能体的轨迹
        for i in range(self.n):
            ax1.plot(t, [i] * len(t), states[i, 0, :] * 1000, linewidth=2, color=colors[i], label=f'EHA {i + 1}')

        # 确保x轴从0开始
        ax1.set_xlim(left=0, right=max(t))

        # 设置坐标轴标签
        ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold', labelpad=10)
        ax1.set_ylabel('Agent ID', fontsize=12, fontweight='bold', labelpad=10)
        ax1.set_zlabel('Displacement (mm)', fontsize=12, fontweight='bold', labelpad=15)  # 修复竖轴标签

        ax1.set_title('3D Displacement Tracking Performance', fontsize=16, fontweight='bold', pad=5)
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)
        ax1.view_init(elev=25, azim=45)  # 设置视角

        # 调整3D图的边距以防止标签被裁剪
        plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95)

        # 保存图1 - 使用不同的bbox_inches参数
        fig1.savefig(f'EHA_fig2/fig1_3d_displacement.pdf', dpi=300, bbox_inches=None, pad_inches=0.5)
        fig1.savefig(f'EHA_fig2/fig1_3d_displacement.png', dpi=300, bbox_inches=None, pad_inches=0.5)
        fig1.savefig(f'EHA_fig2/fig1_3d_displacement.svg', dpi=300, bbox_inches=None, pad_inches=0.5)
        plt.show(block=False)

        # ========== 图表2: 速度响应 ==========
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for i in range(self.n):
            ax2.plot(t, states[i, 1, :], linewidth=2, color=colors[i], label=f'EHA {i + 1}')
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Velocity (m/s)', fontsize=12)
        ax2.set_title('Velocity Response', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        ax2.set_ylim(-0.03, 0.01)     # 新增：设置纵轴显示范围
        plt.tight_layout()
        plt.savefig('EHA_fig2/fig2_eha_velocity.png', dpi=300, bbox_inches='tight')
        plt.savefig('EHA_fig2/fig2_eha_velocity.svg', format='svg', bbox_inches='tight')
        plt.savefig('EHA_fig2/fig2_eha_velocity.pdf', dpi=300, bbox_inches='tight')
        plt.show(block=False)

        # ========== 图表3: 压力响应 ==========
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        for i in range(self.n):
            ax3.plot(t, states[i, 2, :] / 1e6, linewidth=2, color=colors[i], label=f'EHA {i + 1}')
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('Pressure (MPa)', fontsize=12)
        ax3.set_title('Load Pressure Response', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=10)
        ax3.set_ylim(0, 2.5)  # 新增：设置纵轴显示范围
        plt.tight_layout()
        plt.savefig('EHA_fig2/fig3_eha_pressure.png', dpi=300, bbox_inches='tight')
        plt.savefig('EHA_fig2/fig3_eha_pressure.svg', format='svg', bbox_inches='tight')
        plt.savefig('EHA_fig2/fig3_eha_pressure.pdf', dpi=300, bbox_inches='tight')
        plt.show(block=False)

        # ========== 图表4: 控制电压 ==========
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        for i in range(self.n):
            ax4.plot(t, v_history[i, :], linewidth=2, color=colors[i], label=f'EHA {i + 1}')
        ax4.set_xlabel('Time (s)', fontsize=12)
        ax4.set_ylabel('Control Voltage (V)', fontsize=12)
        ax4.set_title('Control Voltage Input', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best', fontsize=10)
        ax4.set_ylim(-0.2, 0.075)  # 新增：设置纵轴显示范围
        plt.tight_layout()
        plt.savefig('EHA_fig2/fig4_eha_voltage.png', dpi=300, bbox_inches='tight')
        plt.savefig('EHA_fig2/fig4_eha_voltage.svg', format='svg', bbox_inches='tight')
        plt.savefig('EHA_fig2/fig4_eha_voltage.pdf', dpi=300, bbox_inches='tight')
        plt.show(block=False)

        # ========== 图表5: 跟踪误差 ==========
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        for i in range(self.n):
            ax5.plot(t, tracking_errors[i, :] * 1000, linewidth=2, color=colors[i], label=f'EHA {i + 1}')

        # 添加钢轨打磨精度区域
        ax5.axhspan(-0.005, 0.005, alpha=0.2, color='blue', label='Rail Grinding Precision (±0.01mm)')
        ax5.set_xlabel('Time (s)', fontsize=12)
        ax5.set_ylabel('Tracking Error (mm)', fontsize=12)
        ax5.set_title('Tracking Error', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig('EHA_fig2/fig5_eha_tracking_error.png', dpi=300, bbox_inches='tight')
        plt.savefig('EHA_fig2/fig5_eha_tracking_error.svg', format='svg', bbox_inches='tight')
        plt.savefig('EHA_fig2/fig5_eha_tracking_error.pdf', dpi=300, bbox_inches='tight')
        plt.show(block=False)

        # ========== 图表6: 成本函数 ==========
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        for i in range(self.n):
            ax6.plot(t, cost_functions[i, :], linewidth=2, color=colors[i], label=f'EHA {i + 1}')
        ax6.set_xlabel('Time (s)', fontsize=12)
        ax6.set_ylabel('Cost Function', fontsize=12)
        ax6.set_title('Cost Function', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='best', fontsize=10)
        ax6.set_ylim(-1000, 100)  # 新增：设置纵轴显示范围
        plt.tight_layout()
        plt.savefig('EHA_fig2/fig6_eha_cost_function.png', dpi=300, bbox_inches='tight')
        plt.savefig('EHA_fig2/fig6_eha_cost_function.svg', format='svg', bbox_inches='tight')
        plt.savefig('EHA_fig2/fig6_eha_cost_function.pdf', dpi=300, bbox_inches='tight')
        plt.show(block=False)

        # ========== 图表7: 第三步Critic权重范数 ==========
        fig7, ax7 = plt.subplots(figsize=(6, 4))
        for i in range(self.n):
            ax7.plot(t, Mc_norms[i, :], linewidth=2, color=colors[i], label=f'EHA {i + 1}')
        ax7.set_xlabel('Time (s)', fontsize=12)
        ax7.set_ylabel(r'$\|W_c\|$', fontsize=14)
        ax7.set_title('Critic NN Weight Norm (Step 3)', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig('EHA_fig2/fig7_eha_critic_norm.png', dpi=300, bbox_inches='tight')
        plt.savefig('EHA_fig2/fig7_eha_critic_norm.svg', format='svg', bbox_inches='tight')
        plt.savefig('EHA_fig2/fig7_eha_critic_norm.pdf', dpi=300, bbox_inches='tight')
        plt.show(block=False)

        # ========== 图表8: 第三步Actor权重范数 ==========
        fig8, ax8 = plt.subplots(figsize=(6, 4))
        for i in range(self.n):
            ax8.plot(t, Ma_norms[i, :], linewidth=2, color=colors[i], label=f'EHA {i + 1}')
        ax8.set_xlabel('Time (s)', fontsize=12)
        ax8.set_ylabel(r'$\|W_a\|$', fontsize=14)
        ax8.set_title('Actor NN Weight Norm (Step 3)', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig('EHA_fig2/fig8_eha_actor_norm.png', dpi=300, bbox_inches='tight')
        plt.savefig('EHA_fig2/fig8_eha_actor_norm.svg', format='svg', bbox_inches='tight')
        plt.savefig('EHA_fig2/fig8_eha_actor_norm.pdf', dpi=300, bbox_inches='tight')
        plt.show(block=False)

        # ========== 图表9: 第二步Critic权重范数 ==========
        fig9, ax9 = plt.subplots(figsize=(6, 4))
        for i in range(self.n):
            ax9.plot(t, Mc2_norms[i, :], linewidth=2, color=colors[i], label=f'EHA {i + 1}')
        ax9.set_xlabel('Time (s)', fontsize=12)
        ax9.set_ylabel(r'$\|W_{c2}\|$', fontsize=14)
        ax9.set_title('Critic NN Weight Norm (Step 2)', fontsize=14, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        ax9.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig('EHA_fig2/fig9_eha_critic2_norm.png', dpi=300, bbox_inches='tight')
        plt.savefig('EHA_fig2/fig9_eha_critic2_norm.svg', format='svg', bbox_inches='tight')
        plt.savefig('EHA_fig2/fig9_eha_critic2_norm.pdf', dpi=300, bbox_inches='tight')
        plt.show(block=False)

        # ========== 图表10: 第二步Actor权重范数 ==========
        fig10, ax10 = plt.subplots(figsize=(6, 4))
        for i in range(self.n):
            ax10.plot(t, Ma2_norms[i, :], linewidth=2, color=colors[i], label=f'EHA {i + 1}')
        ax10.set_xlabel('Time (s)', fontsize=12)
        ax10.set_ylabel(r'$\|W_{a2}\|$', fontsize=14)
        ax10.set_title('Actor NN Weight Norm (Step 2)', fontsize=14, fontweight='bold')
        ax10.grid(True, alpha=0.3)
        ax10.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig('EHA_fig2/fig10_eha_actor2_norm.png', dpi=300, bbox_inches='tight')
        plt.savefig('EHA_fig2/fig10_eha_actor2_norm.svg', format='svg', bbox_inches='tight')
        plt.savefig('EHA_fig2/fig10_eha_actor2_norm.pdf', dpi=300, bbox_inches='tight')
        plt.show(block=False)

        # ========== 图表11: 第二步扰动权重范数 ==========
        fig11, ax11 = plt.subplots(figsize=(6, 4))
        for i in range(self.n):
            ax11.plot(t, Md2_norms[i, :], linewidth=2, color=colors[i], label=f'EHA {i + 1}')
        ax11.set_xlabel('Time (s)', fontsize=12)
        ax11.set_ylabel(r'$\|W_{d2}\|$', fontsize=14)
        ax11.set_title('Disturbance NN Weight Norm (Step 2)', fontsize=14, fontweight='bold')
        ax11.grid(True, alpha=0.3)
        ax11.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig('EHA_fig2/fig11_eha_disturbance2_norm.png', dpi=300, bbox_inches='tight')
        plt.savefig('EHA_fig2/fig11_eha_disturbance2_norm.svg', format='svg', bbox_inches='tight')
        plt.savefig('EHA_fig2/fig11_eha_disturbance2_norm.pdf', dpi=300, bbox_inches='tight')
        plt.show(block=False)

        # ========== 图表12: 第三步扰动权重范数 ==========
        fig12, ax12 = plt.subplots(figsize=(6, 4))
        for i in range(self.n):
            ax12.plot(t, Md_norms[i, :], linewidth=2, color=colors[i], label=f'EHA {i + 1}')
        ax12.set_xlabel('Time (s)', fontsize=12)
        ax12.set_ylabel(r'$\|W_{d}\|$', fontsize=14)
        ax12.set_title('Disturbance NN Weight Norm (Step 3)', fontsize=14, fontweight='bold')
        ax12.grid(True, alpha=0.3)
        ax12.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig('EHA_fig2/fig12_eha_disturbance_norm.png', dpi=300, bbox_inches='tight')
        plt.savefig('EHA_fig2/fig12_eha_disturbance_norm.svg', format='svg', bbox_inches='tight')
        plt.savefig('EHA_fig2/fig12_eha_disturbance_norm.pdf', dpi=300, bbox_inches='tight')
        plt.show(block=False)

        # ========== 图表13: 第二步扰动观测值 ==========
        fig13, ax13 = plt.subplots(figsize=(6, 4))
        for i in range(self.n):
            ax13.plot(t, d2_history[i, :], linewidth=2, color=colors[i], label=f'EHA {i + 1}')
        ax13.set_xlabel('Time (s)', fontsize=12)
        ax13.set_ylabel(r'$\hat{d}_2$', fontsize=14)
        ax13.set_title('Disturbance Estimate (Step 2)', fontsize=14, fontweight='bold')
        ax13.grid(True, alpha=0.3)
        ax13.legend(loc='best', fontsize=10)
        ax13.set_ylim(-5, 10)  # 新增：设置纵轴显示范围
        plt.tight_layout()
        plt.savefig('EHA_fig2/fig13_eha_disturbance2_estimate.png', dpi=300, bbox_inches='tight')
        plt.savefig('EHA_fig2/fig13_eha_disturbance2_estimate.svg', format='svg', bbox_inches='tight')
        plt.show(block=False)

        # ========== 图表14: 第三步扰动观测值 ==========
        fig14, ax14 = plt.subplots(figsize=(6, 4))
        for i in range(self.n):
            ax14.plot(t, d_history[i, :], linewidth=2, color=colors[i], label=f'EHA {i + 1}')
        ax14.set_xlabel('Time (s)', fontsize=12)
        ax14.set_ylabel(r'$\hat{d}_3$', fontsize=14)
        ax14.set_title('Disturbance Estimate (Step 3)', fontsize=14, fontweight='bold')
        ax14.grid(True, alpha=0.3)
        ax14.legend(loc='best', fontsize=10)
        ax14.set_ylim(-500, 1600)  # 新增：设置纵轴显示范围
        plt.tight_layout()
        plt.savefig('EHA_fig2/fig14_eha_disturbance_estimate.png', dpi=300, bbox_inches='tight')
        plt.savefig('EHA_fig2/fig14_eha_disturbance_estimate.svg', format='svg', bbox_inches='tight')
        plt.show(block=False)

        # 性能分析
        print("\nEHA系统仿真结果分析:")
        print(f"仿真时长: {t[-1]:.2f} 秒")
        print(f"最终跟踪误差 (RMS):")
        for i in range(self.n):
            rms_error = np.sqrt(np.mean(tracking_errors[i, -100:] ** 2))
            print(f"  智能体{i + 1}: {rms_error * 1000:.5f} mm")

        # 一致性误差分析
        final_outputs = states[:, 0, -1]
        consensus_error = np.std(final_outputs)
        print(f"最终一致性误差 (标准差): {consensus_error * 1000:.5f} mm")

        # 扰动估计性能分析
        print(f"\n扰动估计性能:")
        for i in range(self.n):
            mean_d2 = np.mean(d2_history[i, -100:])
            mean_d3 = np.mean(d_history[i, -100:])
            print(f"  智能体{i + 1}: 第二步估计={mean_d2:.4f}, 第三步估计={mean_d3:.4f}")

        plt.show()# 保证窗口不会关闭


def main():
    """主函数"""
    print("=" * 70)
    print("EHA电液伺服系统多智能体一致优化控制仿真")
    print("基于反馈线性化的强化学习控制")
    print("第二步反步控制已加入强化学习设计")
    print("=" * 70)

    # 创建EHA多智能体系统
    system = EHAMultiAgentSystem()

    # 运行仿真
    sol = system.simulate()

    # 分析结果
    results = system.analyze_results(sol)

    # 可视化结果
    system.visualize_results(results)

    print("\nEHA系统最优一致性控制仿真完成！")


if __name__ == "__main__":
    main()