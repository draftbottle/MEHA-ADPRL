# optimal_eha_FNO.py
# 重构版 - 保持与原有仿真接口完全兼容
# 实现真正的Farthest Better or Nearest Worse Optimizer (FNO)

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 设置中文字体支持
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']    # 设置学术图表样式
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# 将当前目录添加到路径，以便导入主仿真模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 尝试导入仿真模块
try:
    from MEHA_RL_FNO import EHAMultiAgentSystem

    SIM_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入仿真模块: {e}")
    print("将退出算法迭代")
    SIM_AVAILABLE = False


class FNOptimizer:
    """
    真正的Farthest Better or Nearest Worse Optimizer (FNO) 实现
    严格遵循官方MATLAB代码的逻辑: https://github.com/AhmadTaheri2021/FNO-Optimizer
    """

    def __init__(self, n_agents, dimensions, lower_bounds, upper_bounds, max_iter=50, w=0.9, c1=0.5, c2=0.5):
        """
        初始化FNO优化器
        注意: 为了保持接口兼容，保留了w, c1, c2参数，但内部不使用PSO逻辑

        :param n_agents: 种群大小（智能体数量）
        :param dimensions: 优化问题的维度
        :param lower_bounds: 每个维度的下界，一维数组
        :param upper_bounds: 每个维度的上界，一维数组
        :param max_iter: 最大迭代次数
        :param w: 惯性权重 (保留参数，内部不使用)
        :param c1: 个体认知参数 (保留参数，内部不使用)
        :param c2: 社会认知参数 (保留参数，内部不使用)
        """
        self.n_agents = n_agents
        self.dim = dimensions
        self.lb = np.array(lower_bounds)
        self.ub = np.array(upper_bounds)
        self.max_iter = max_iter

        # 注意: w, c1, c2参数保留但不使用，以保持接口兼容

        # 初始化种群位置
        self.positions = np.random.uniform(self.lb, self.ub, (self.n_agents, self.dim))

        # 存储每个智能体的适应度
        self.fitness_values = np.full(self.n_agents, np.inf)

        # 全局最优
        self.gbest_position = None
        self.gbest_fitness = np.inf

        # 记录迭代历史
        self.history_best_fitness = []
        self.history_mean_fitness = []
        self.history_worst_fitness = []
        self.history_std_fitness = []

        # 收敛统计
        self.convergence_iteration = 0
        self.convergence_threshold = 1e-6
        self.stagnation_count = 0

    def _calculate_distances(self, positions):
        """计算所有个体之间的欧氏距离矩阵"""
        n = len(positions)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def _find_reference_individuals(self, positions, fitness_values):
        """
        为每个个体找到FB(最远更好)和NW(最近更差)个体
        严格遵循官方MATLAB代码逻辑
        """
        n = len(positions)

        # 1. 按适应度排序（最小为最优）
        sorted_indices = np.argsort(fitness_values)

        # 2. 计算距离矩阵
        distances = self._calculate_distances(positions)

        # 3. 初始化FB和NW数组
        FB = np.arange(n)  # 初始化为自身
        NW = np.arange(n)  # 初始化为自身

        for i in range(n):
            # 找到个体i在排序中的位置
            pos_rank = np.where(sorted_indices == i)[0][0]

            # 划分更好和更差的个体集合
            if pos_rank == 0:  # 当前是最优个体
                betters = np.array([])
                worses = sorted_indices[pos_rank + 1:]
            elif pos_rank == n - 1:  # 当前是最差个体
                betters = sorted_indices[:pos_rank]
                worses = np.array([])
            else:
                betters = sorted_indices[:pos_rank]
                worses = sorted_indices[pos_rank + 1:]

            # 动态调整更好集合的大小（官方代码中的(1-(FEs/MaxFEs)^0.5) * rand）
            if len(betters) > 0:
                # 模拟迭代过程的影响
                alpha = 0.5  # 初始值，实际优化时会随迭代变化
                keep_ratio = (1 - alpha ** 0.5) * np.random.rand()
                keep_count = int(np.ceil(len(betters) * keep_ratio))
                if keep_count < len(betters):
                    betters = betters[:keep_count]

            # 找到最远更好(FB)个体
            if len(betters) > 0:
                # 在更好的个体中找距离最远的
                dist_to_betters = distances[i, betters]
                farthest_idx = np.argmax(dist_to_betters)
                FB[i] = betters[farthest_idx]

            # 找到最近更差(NW)个体
            if len(worses) > 0:
                # 在更差的个体中找距离最近的
                dist_to_worses = distances[i, worses]
                nearest_idx = np.argmin(dist_to_worses)
                NW[i] = worses[nearest_idx]

        return FB, NW

    def _dynamic_focus_strategy(self, iteration):
        """
        动态聚焦策略(DFS)
        官方公式: DFF = abs(unifrnd((Mu-alpha), (Mu+alpha), VarSize))
        其中 alpha = (1-(FEs/MaxFEs))^(1/2)
        """
        # 计算alpha，随迭代递减
        alpha = (1 - (iteration / self.max_iter)) ** 0.5

        # 生成Mu
        Mu = np.random.rand()

        # 生成动态聚焦因子DFF
        DFF = np.abs(np.random.uniform(Mu - alpha, Mu + alpha, self.dim))

        return DFF, alpha

    def _update_position(self, i, positions, fitness_values, FB, NW, iteration):
        """更新个体位置，严格遵循官方MATLAB代码逻辑"""
        new_pos = positions[i].copy()

        # 获取参考个体
        fb_pos = positions[FB[i]]
        nw_pos = positions[NW[i]]
        current_pos = positions[i]

        # ----------------------- 计算S因子 -------------------------
        # 官方代码: S = max(0, sign(Position(FB(i)).Position - Position(i).Position) .*
        #                  sign(Position(NW(i)).Position - Position(i).Position))
        S = np.maximum(0, np.sign(fb_pos - current_pos) * np.sign(nw_pos - current_pos))

        # ----------------------- 计算临时位置Pos_temp -------------------------
        # 官方代码核心部分
        term1 = S * nw_pos
        term2 = (1 - S) * current_pos
        term3 = np.sign(fb_pos - current_pos) * np.random.rand(self.dim) * np.abs(nw_pos - current_pos)

        Pos_temp = term1 + term2 + term3

        # ----------------------- 应用动态聚焦策略 -------------------------
        DFF, alpha = self._dynamic_focus_strategy(iteration)

        # 计算W因子（官方代码: W = (2 + unifrnd(-1,+1))*randi([1 2])）
        W = (2 + np.random.uniform(-1, 1)) * np.random.choice([1, 2])

        # 以概率决定使用哪种策略
        # 官方代码: if rand < 1-(FEs/MaxFEs) 则用Pos_temp，否则用DFS
        if np.random.rand() < 1 - (iteration / self.max_iter):
            # 策略1: 直接使用Pos_temp（跳过或逃离NW）
            new_pos = Pos_temp
        else:
            # 策略2: 应用DFS探索FB
            # 官方代码: Exp = DFF .* (Position(FB(i)).Position - Pos_temp)
            #         newPos.Position = Pos_temp + W .* Exp
            Exp = DFF * (fb_pos - Pos_temp)
            new_pos = Pos_temp + W * Exp

        # ----------------------- 边界处理 -------------------------
        # 官方使用Clipping函数，这里实现类似逻辑
        under_lb = new_pos < self.lb
        over_ub = new_pos > self.ub

        if np.any(under_lb):
            new_pos[under_lb] = self.lb[under_lb] + np.random.rand(np.sum(under_lb)) * (
                        self.ub[under_lb] - self.lb[under_lb])

        if np.any(over_ub):
            new_pos[over_ub] = self.lb[over_ub] + np.random.rand(np.sum(over_ub)) * (
                        self.ub[over_ub] - self.lb[over_ub])

        return new_pos

    def _initialize_fitness(self, fitness_func):
        """初始化种群适应度"""
        for i in range(self.n_agents):
            self.fitness_values[i] = fitness_func(self.positions[i])

            # 更新全局最优
            if self.fitness_values[i] < self.gbest_fitness:
                self.gbest_fitness = self.fitness_values[i]
                self.gbest_position = self.positions[i].copy()

    def _print_iteration_info(self, iter, fitness_values, verbose_level=1):
        """打印迭代信息"""
        # 计算统计信息
        best_fitness = np.min(fitness_values)
        mean_fitness = np.mean(fitness_values)
        worst_fitness = np.max(fitness_values)
        std_fitness = np.std(fitness_values)
        progress = (iter + 1) / self.max_iter * 100

        if verbose_level >= 1:
            # 基本迭代信息
            print(f"\n[迭代 {iter + 1:3d}/{self.max_iter}] 进度: {progress:5.1f}%")

            # 适应度统计
            print(f"   适应度统计: 最佳={best_fitness:10.6f} | "
                  f"平均={mean_fitness:10.6f} | "
                  f"最差={worst_fitness:10.6f} | "
                  f"标准差={std_fitness:10.6f}")

            # 收敛检测
            if len(self.history_best_fitness) > 10:
                improvement = self.history_best_fitness[-10] - self.gbest_fitness
                if abs(improvement) < self.convergence_threshold:
                    self.stagnation_count += 1
                    if self.stagnation_count > 5:
                        print(f"   注意: 算法已连续{self.stagnation_count}次迭代没有显著改进")
                else:
                    self.stagnation_count = 0

        if verbose_level >= 2 and (iter + 1) % 5 == 0:
            # 每5次迭代显示详细参数
            print(f"   当前最优参数:")
            param_names = ['gamma1', 'gamma2', 'gamma3', 'kappa_c', 'kappa_a', 'kappa_d', 'sigma', 'rho']
            for i, (name, value) in enumerate(zip(param_names, self.gbest_position)):
                print(f"     {name}: {value:10.6f}")

    def optimize(self, fitness_func, verbose=True):
        """
        执行FNO优化

        :param fitness_func: 适应度函数，接受一个参数向量，返回一个标量适应度值
        :param verbose: 是否打印优化过程信息
        :return: 最优适应度值，最优参数向量
        """
        # 初始化适应度
        self._initialize_fitness(fitness_func)

        if verbose:
            print("\n" + "=" * 80)
            print("真正的FNO优化器初始化完成")
            print("=" * 80)
            print(f"种群大小: {self.n_agents}")
            print(f"优化维度: {self.dim}")
            print(f"最大迭代次数: {self.max_iter}")
            print(f"初始最优适应度: {self.gbest_fitness:.6f}")
            print(f"初始最优参数: {self.gbest_position.round(4)}")
            print("=" * 80)

        # 主优化循环
        for iteration in range(self.max_iter):
            # 步骤1: 为每个个体找到FB和NW参考个体
            FB, NW = self._find_reference_individuals(self.positions, self.fitness_values)

            # 步骤2: 更新每个个体的位置
            new_positions = np.zeros_like(self.positions)
            new_fitness_values = np.zeros_like(self.fitness_values)

            for i in range(self.n_agents):
                # 生成新位置
                new_pos = self._update_position(i, self.positions, self.fitness_values, FB, NW, iteration)
                new_positions[i] = new_pos

                # 评估新位置
                new_fitness = fitness_func(new_pos)
                new_fitness_values[i] = new_fitness

                # 贪婪选择：只有更好的解才被接受
                if new_fitness < self.fitness_values[i]:
                    self.positions[i] = new_pos
                    self.fitness_values[i] = new_fitness

                    # 更新全局最优
                    if new_fitness < self.gbest_fitness:
                        self.gbest_fitness = new_fitness
                        self.gbest_position = new_pos.copy()
                        self.convergence_iteration = iteration + 1
                else:
                    # 保持原位置
                    new_positions[i] = self.positions[i]
                    new_fitness_values[i] = self.fitness_values[i]

            # 记录本次迭代统计
            self.history_best_fitness.append(self.gbest_fitness)
            self.history_mean_fitness.append(np.mean(self.fitness_values))
            self.history_worst_fitness.append(np.max(self.fitness_values))
            self.history_std_fitness.append(np.std(self.fitness_values))

            # 显示迭代信息
            if verbose:
                verbose_level = 2 if (iteration + 1) % 5 == 0 or iteration == 0 or iteration == self.max_iter - 1 else 1
                self._print_iteration_info(iteration, self.fitness_values, verbose_level)

        if verbose:
            self._print_optimization_summary()

        return self.gbest_fitness, self.gbest_position

    def _print_optimization_summary(self):
        """打印优化总结"""
        print("\n" + "=" * 80)
        print("真正的FNO优化完成 - 总结报告")
        print("=" * 80)
        print(f"总迭代次数: {self.max_iter}")
        print(f"收敛于第 {self.convergence_iteration} 次迭代")
        print(f"最终最优适应度: {self.gbest_fitness:.6f}")

        if len(self.history_best_fitness) > 0:
            initial_fitness = self.history_best_fitness[0]
            improvement = initial_fitness - self.gbest_fitness
            improvement_percent = (improvement / initial_fitness * 100) if initial_fitness != 0 else 0
            print(f"适应度改进: {improvement:.6f} ({improvement_percent:.1f}%)")

        print("-" * 80)

        # 收敛分析
        if len(self.history_best_fitness) > 1:
            initial_improvement = self.history_best_fitness[0] - self.history_best_fitness[10] if len(
                self.history_best_fitness) > 10 else 0
            final_improvement = self.history_best_fitness[-10] - self.history_best_fitness[-1] if len(
                self.history_best_fitness) > 10 else 0

            print("收敛分析:")
            print(f"  前10次迭代改进: {initial_improvement:.6f}")
            print(f"  后10次迭代改进: {final_improvement:.6f}")

            if final_improvement < self.convergence_threshold * 10:
                print("  状态: 算法已收敛到稳定解")
            else:
                print("  状态: 算法仍在改进中，可考虑增加迭代次数")

        print("=" * 80)

    def plot_convergence(self, save_path="FNO_convergence.png"):
        """绘制收敛曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        iterations = range(1, len(self.history_best_fitness) + 1)

        # 子图1: 最佳适应度
        ax1 = axes[0, 0]
        ax1.plot(iterations, self.history_best_fitness, 'b-', linewidth=2, label='最佳适应度')
        ax1.set_xlabel('迭代次数', fontsize=12)
        ax1.set_ylabel('最佳适应度', fontsize=12)
        ax1.set_title('最佳适应度收敛曲线', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_yscale('log')

        # 子图2: 平均适应度
        ax2 = axes[0, 1]
        ax2.plot(iterations, self.history_mean_fitness, 'r-', linewidth=2, label='平均适应度')
        ax2.set_xlabel('迭代次数', fontsize=12)
        ax2.set_ylabel('平均适应度', fontsize=12)
        ax2.set_title('平均适应度变化', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_yscale('log')

        # 子图3: 最佳vs最差适应度
        ax3 = axes[1, 0]
        ax3.plot(iterations, self.history_best_fitness, 'b-', linewidth=2, label='最佳适应度')
        ax3.plot(iterations, self.history_worst_fitness, 'r-', linewidth=2, label='最差适应度')
        ax3.fill_between(iterations, self.history_best_fitness, self.history_worst_fitness,
                         alpha=0.2, color='gray', label='适应度范围')
        ax3.set_xlabel('迭代次数', fontsize=12)
        ax3.set_ylabel('适应度', fontsize=12)
        ax3.set_title('适应度范围变化', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        ax3.set_yscale('log')

        # 子图4: 适应度标准差
        ax4 = axes[1, 1]
        ax4.plot(iterations, self.history_std_fitness, 'g-', linewidth=2, label='适应度标准差')
        ax4.set_xlabel('迭代次数', fontsize=12)
        ax4.set_ylabel('标准差', fontsize=12)
        ax4.set_title('种群多样性变化', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)

        plt.suptitle('真正的FNO优化算法收敛分析', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def objective_function(params, verbose=False):
    """
    目标函数：计算一组参数的适应度（总绝对跟踪误差）
    params: 一个参数向量 [gamma1, gamma2, gamma3, kappa_c, kappa_a, kappa_d, sigma, rho]
    注意: 此函数与原有代码中的objective_function完全兼容
    """
    # 1. 从参数向量解析各个参数 (现在有8个参数)
    gamma1, gamma2, gamma3, kappa_c, kappa_a, kappa_d, sigma, rho = params

    # 2. 如果仿真模块可用，使用真实的仿真接口
    if SIM_AVAILABLE:
        try:
            system = EHAMultiAgentSystem()

            # 遍历所有agent，应用新的参数
            for agent in system.agents:
                agent.gamma1 = gamma1
                agent.gamma2 = gamma2
                agent.gamma3 = gamma3
                agent.kappa_c = kappa_c
                agent.kappa_a = kappa_a
                agent.kappa_d = kappa_d
                agent.sigma = sigma
                agent.rho = rho

            # 3. 运行仿真
            sol = system.simulate()
            results = system.analyze_results(sol)

            # 4. 计算适应度：所有智能体、所有时间步的绝对跟踪误差之和
            tracking_errors = results['tracking_errors']
            fitness = np.sum(np.abs(tracking_errors))

            if verbose:
                print(f"参数={params.round(4)}, 适应度={fitness:.6f}")

            return fitness

        except Exception as e:
            # 如果仿真出错（如参数导致不稳定），赋予一个很大的惩罚值
            print(f"参数 {params.round(4)} 仿真失败，错误: {e}，赋予大惩罚值。")
            return 1e10
    else:
        # 仿真模块不可用，使用模拟适应度函数
        # 模拟一个简单的目标函数，实际使用时应该替换为真实仿真
        # target_params = np.array([5.0, 5.0, 5.0, 0.5, 0.5, 0.5, 1.0, 1.0])
        # noise = np.random.randn() * 0.1
        # fitness = np.sum(np.abs(params - target_params)) + np.abs(noise)
        # if verbose:
        #     print(f"参数={params.round(4)}, 模拟适应度={fitness:.6f}")

        return FileNotFoundError    # 仿真不可用则直接报错


def main():
    """主函数：使用真正的FNO优化EHA控制器参数"""
    print("=" * 80)
    print("EHA控制器参数优化 - 使用真正的Farthest better or nearest worse optimizer (FNO)")
    print("包含参数: gamma1, gamma2, gamma3, kappa_c, kappa_a, kappa_d, sigma, rho")
    print("=" * 80)

    if not SIM_AVAILABLE:
        print("警告: 仿真模块不可用，将使用模拟适应度函数")
        print("=" * 80)

    # 1. 设置优化器参数
    n_agents = 20  # 种群大小
    n_iterations = 30  # 迭代次数
    dimensions = 8  # 优化变量的维度，对应8个参数 (新增了kappa_d)

    # 参数边界 [gamma1, gamma2, gamma3, kappa_c, kappa_a, kappa_d, sigma, rho]
    lower_bounds = np.array([50, 50, 50, 1, 1, 0.1, 0.01, 0.5])
    upper_bounds = np.array([500, 500, 500, 100, 100, 10.0, 2.0, 10.0])

    param_names = ['gamma1', 'gamma2', 'gamma3', 'kappa_c', 'kappa_a', 'kappa_d', 'sigma', 'rho']

    print(f"优化维度: {dimensions}")
    print(f"参数范围:")
    for i, (name, lb, ub) in enumerate(zip(param_names, lower_bounds, upper_bounds)):
        print(f"  {name}: [{lb}, {ub}]")
    print(f"种群大小: {n_agents}")
    print(f"最大迭代次数: {n_iterations}")

    # 2. 创建真正的FNO优化器
    print("\n初始化真正的FNO优化器...")
    optimizer = FNOptimizer(
        n_agents=n_agents,
        dimensions=dimensions,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_iter=n_iterations,
        w=0.9,  # 保留参数，内部不使用
        c1=0.5,  # 保留参数，内部不使用
        c2=0.5  # 保留参数，内部不使用
    )

    # 3. 定义适应度函数包装
    def fitness_wrapper(params):
        return objective_function(params, verbose=False)

    # 4. 执行优化
    print("\n开始真正的FNO优化...")
    print("=" * 80)

    best_cost, best_pos = optimizer.optimize(fitness_wrapper, verbose=True)

    # 5. 输出最优结果
    print("\n" + "=" * 80)
    print("真正的FNO优化完成!")
    print("=" * 80)
    print(f"最优适应度值(总绝对跟踪误差): {best_cost:.6f}")
    print(f"最优参数组合:")
    for name, value in zip(param_names, best_pos):
        print(f"  {name}: {value:.6f}")

    # 6. 绘制详细的收敛曲线
    print("\n绘制收敛曲线...")
    optimizer.plot_convergence(save_path="True_FNO_convergence_curve.png")

    # 7. 用最优参数运行一次完整仿真并绘图
    if SIM_AVAILABLE:
        print("\n" + "=" * 80)
        print("使用最优参数进行最终验证仿真...")
        print("=" * 80)

        system_final = EHAMultiAgentSystem()
        for agent in system_final.agents:
            agent.gamma1 = best_pos[0]
            agent.gamma2 = best_pos[1]
            agent.gamma3 = best_pos[2]
            agent.kappa_c = best_pos[3]
            agent.kappa_a = best_pos[4]
            agent.kappa_d = best_pos[5]
            agent.sigma = best_pos[6]
            agent.rho = best_pos[7]

        sol_final = system_final.simulate()
        results_final = system_final.analyze_results(sol_final)

        # 创建新的目录保存结果
        os.makedirs("EHA_fig_True_FNO_optimized", exist_ok=True)
        system_final.visualize_results(results_final)

        # 保存最优参数到文件
        np.savetxt('best_parameters_true_fno.txt', best_pos,
                   header=' '.join(param_names),
                   fmt='%.6f')
        print("\n最优参数已保存到 'best_parameters_true_fno.txt'")
    else:
        print("\n注意: 仿真模块不可用，跳过最终验证仿真步骤")

    # 8. 保存优化配置
    with open('true_fno_optimization_config.txt', 'w') as f:
        f.write("真正的FNO优化配置\n")
        f.write("=" * 50 + "\n")
        f.write(f"种群大小: {n_agents}\n")
        f.write(f"迭代次数: {n_iterations}\n")
        f.write(f"最优适应度: {best_cost:.6f}\n")
        f.write(f"收敛迭代: {optimizer.convergence_iteration}\n")
        f.write("\n参数边界:\n")
        for name, lb, ub in zip(param_names, lower_bounds, upper_bounds):
            f.write(f"  {name}: [{lb}, {ub}]\n")
        f.write("\n最优参数:\n")
        for name, value in zip(param_names, best_pos):
            f.write(f"  {name}: {value:.6f}\n")

    print("优化配置已保存到 'true_fno_optimization_config.txt'")
    print("\n真正的FNO优化完成！")


if __name__ == "__main__":
    main()