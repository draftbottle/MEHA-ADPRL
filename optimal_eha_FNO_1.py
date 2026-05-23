# optimal_eha_fno.py (精简版 - 移除时间计算)
# 使用Farthest better or nearest worse optimizer (FNO) 优化EHA控制器参数
# 替代原有的PSO算法
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 将当前目录添加到路径，以便导入主仿真模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MEHA_RL_FNO import EHAMultiAgentSystem  # 导入您的仿真系统


class FNOptimizer:
    """
    最优或最劣优化器 (Farthest better or nearest worse optimizer, FNO) 实现
    参考自2026年相关论文描述的核心思想。
    """

    def __init__(self, n_agents, dimensions, lower_bounds, upper_bounds, max_iter=50, w=0.9, c1=0.5, c2=0.5):
        """
        初始化FNO优化器
        :param n_agents: 种群大小（智能体数量）
        :param dimensions: 优化问题的维度
        :param lower_bounds: 每个维度的下界，一维数组
        :param upper_bounds: 每个维度的上界，一维数组
        :param max_iter: 最大迭代次数
        :param w: 惯性权重 (用于位置更新)
        :param c1: 个体认知参数 (向自身历史最优学习)
        :param c2: 社会认知参数 (向全局最优学习)
        """
        self.n_agents = n_agents
        self.dim = dimensions
        self.lb = np.array(lower_bounds)
        self.ub = np.array(upper_bounds)
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # 初始化种群位置和速度
        self.positions = np.random.uniform(self.lb, self.ub, (self.n_agents, self.dim))
        self.velocities = np.zeros((self.n_agents, self.dim))

        # 存储每个智能体的历史最优位置和适应度
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.full(self.n_agents, np.inf)

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

    def _initialize_fitness(self, fitness_func):
        """初始化种群适应度"""
        for i in range(self.n_agents):
            fitness = fitness_func(self.positions[i])
            self.pbest_fitness[i] = fitness
            if fitness < self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = self.positions[i].copy()

    def _calculate_distances(self):
        """计算智能体之间的距离矩阵"""
        distances = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances

    def _find_farthest_better(self, agent_idx, fitness_values, distances):
        """找到适应度更好且距离最远的智能体"""
        better_indices = np.where(fitness_values < fitness_values[agent_idx])[0]
        if len(better_indices) > 0:
            # 找到距离最远的更好的智能体
            farthest_idx = better_indices[np.argmax(distances[agent_idx, better_indices])]
            return farthest_idx
        return None

    def _find_nearest_worse(self, agent_idx, fitness_values, distances):
        """找到适应度更差且距离最近的智能体"""
        worse_indices = np.where(fitness_values > fitness_values[agent_idx])[0]
        if len(worse_indices) > 0:
            # 找到距离最近的更差的智能体
            nearest_idx = worse_indices[np.argmin(distances[agent_idx, worse_indices])]
            return nearest_idx
        return None

    def _dynamic_focus_strategy(self, iteration):
        """动态聚焦策略：随迭代调整探索率"""
        # 使用线性递减的随机向量控制探索率
        r = np.random.rand()
        epsilon = 0.1 + 0.8 * (1 - iteration / self.max_iter)  # 从0.9线性递减到0.1
        return r, epsilon

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
            print("FNO优化器初始化完成")
            print("=" * 80)
            print(f"种群大小: {self.n_agents}")
            print(f"优化维度: {self.dim}")
            print(f"最大迭代次数: {self.max_iter}")
            print(f"初始最优适应度: {self.gbest_fitness:.6f}")
            print(f"初始最优参数: {self.gbest_position.round(4)}")
            print("=" * 80)

        # 主优化循环
        for iter in range(self.max_iter):
            # 计算当前所有智能体的适应度
            fitness_values = np.array([fitness_func(pos) for pos in self.positions])

            # 计算距离矩阵
            distances = self._calculate_distances()

            # 应用动态聚焦策略
            r, epsilon = self._dynamic_focus_strategy(iter)

            # 第一阶段：跳过最差的最近区域
            for i in range(self.n_agents):
                nearest_worse = self._find_nearest_worse(i, fitness_values, distances)

                if nearest_worse is not None:
                    # 计算远离最近更差解的方向
                    direction_away = self.positions[i] - self.positions[nearest_worse]
                    # 添加随机扰动避免陷入局部
                    random_vector = np.random.randn(self.dim) * 0.1
                    self.velocities[i] = (self.w * self.velocities[i] +
                                          self.c1 * r * direction_away +
                                          self.c2 * (1 - r) * random_vector)

            # 第二阶段：探索最远的更好区域
            for i in range(self.n_agents):
                farthest_better = self._find_farthest_better(i, fitness_values, distances)

                if farthest_better is not None:
                    # 计算朝向最远更好解的方向
                    direction_toward = self.positions[farthest_better] - self.positions[i]
                    # 使用动态聚焦策略调整步长
                    focus_factor = epsilon * (1 - iter / self.max_iter)
                    self.velocities[i] = (self.w * self.velocities[i] +
                                          self.c1 * focus_factor * direction_toward)

            # 更新位置
            self.positions += self.velocities

            # 确保位置在边界内
            self.positions = np.clip(self.positions, self.lb, self.ub)

            # 评估新位置并更新最优解
            for i in range(self.n_agents):
                fitness = fitness_func(self.positions[i])

                # 更新个体历史最优
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()

                # 更新全局最优
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[i].copy()
                    self.convergence_iteration = iter + 1

            # 记录历史
            self.history_best_fitness.append(self.gbest_fitness)
            self.history_mean_fitness.append(np.mean(fitness_values))
            self.history_worst_fitness.append(np.max(fitness_values))
            self.history_std_fitness.append(np.std(fitness_values))

            # 显示迭代信息
            if verbose:
                verbose_level = 2 if (iter + 1) % 5 == 0 or iter == 0 or iter == self.max_iter - 1 else 1
                self._print_iteration_info(iter, fitness_values, verbose_level)

        if verbose:
            self._print_optimization_summary()

        return self.gbest_fitness, self.gbest_position

    def _print_optimization_summary(self):
        """打印优化总结"""
        print("\n" + "=" * 80)
        print("FNO优化完成 - 总结报告")
        print("=" * 80)
        print(f"总迭代次数: {self.max_iter}")
        print(f"收敛于第 {self.convergence_iteration} 次迭代")
        print(f"最终最优适应度: {self.gbest_fitness:.6f}")
        print(f"适应度改进: {self.history_best_fitness[0] - self.gbest_fitness:.6f} "
              f"({((self.history_best_fitness[0] - self.gbest_fitness) / self.history_best_fitness[0] * 100):.1f}%)")
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

        plt.suptitle('FNO优化算法收敛分析', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def objective_function(params, verbose=False):
    """
    目标函数：计算一组参数的适应度（总绝对跟踪误差）
    params: 一个参数向量 [gamma1, gamma2, gamma3, kappa_c, kappa_a, kappa_d, sigma, rho]
    """
    # 1. 从参数向量解析各个参数 (现在有8个参数)
    gamma1, gamma2, gamma3, kappa_c, kappa_a, kappa_d, sigma, rho = params

    # 2. 使用这组参数创建并运行仿真系统
    try:
        system = EHAMultiAgentSystem()

        # 遍历所有agent，应用新的参数
        for agent in system.agents:
            agent.gamma1 = gamma1
            agent.gamma2 = gamma2
            agent.gamma3 = gamma3
            agent.kappa_c = kappa_c
            agent.kappa_a = kappa_a
            agent.kappa_d = kappa_d  # 新增：设置kappa_d参数
            agent.sigma = sigma
            agent.rho = rho
            # 注意：RBF中心centers依赖于rho和q，这里保持原样

        # 3. 运行仿真
        sol = system.simulate()
        results = system.analyze_results(sol)

        # 4. 计算适应度：所有智能体、所有时间步的绝对跟踪误差之和
        # tracking_errors 形状为 (n_agents, n_time_steps)
        tracking_errors = results['tracking_errors']
        fitness = np.sum(np.abs(tracking_errors))  # ITAE的一种近似

        if verbose:
            print(f"参数={params.round(4)}, 适应度={fitness:.6f}")

        return fitness

    except Exception as e:
        # 如果仿真出错（如参数导致不稳定），赋予一个很大的惩罚值
        print(f"参数 {params.round(4)} 仿真失败，错误: {e}，赋予大惩罚值。")
        return 1e10


def main():
    """主函数：使用FNO优化EHA控制器参数"""
    print("=" * 80)
    print("EHA控制器参数优化 - 使用Farthest better or nearest worse optimizer (FNO)")
    print("包含参数: gamma1, gamma2, gamma3, kappa_c, kappa_a, kappa_d, sigma, rho")
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

    # 2. 创建FNO优化器
    print("\n初始化FNO优化器...")
    optimizer = FNOptimizer(
        n_agents=n_agents,
        dimensions=dimensions,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_iter=n_iterations,
        w=0.9,  # 惯性权重
        c1=0.5,  # 个体认知参数
        c2=0.5  # 社会认知参数
    )

    # 3. 定义适应度函数包装
    def fitness_wrapper(params):
        return objective_function(params, verbose=False)

    # 4. 执行优化
    print("\n开始FNO优化...")
    print("=" * 80)

    best_cost, best_pos = optimizer.optimize(fitness_wrapper, verbose=True)

    # 5. 输出最优结果
    print("\n" + "=" * 80)
    print("优化完成!")
    print("=" * 80)
    print(f"最优适应度值(总绝对跟踪误差): {best_cost:.6f}")
    print(f"最优参数组合:")
    for name, value in zip(param_names, best_pos):
        print(f"  {name}: {value:.6f}")

    # 6. 绘制详细的收敛曲线
    print("\n绘制收敛曲线...")
    optimizer.plot_convergence(save_path="FNO_convergence_curve_with_kappa_d.png")

    # 7. 用最优参数运行一次完整仿真并绘图
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
        agent.kappa_d = best_pos[5]  # 新增：设置优化后的kappa_d
        agent.sigma = best_pos[6]
        agent.rho = best_pos[7]

    sol_final = system_final.simulate()
    results_final = system_final.analyze_results(sol_final)

    # 创建新的目录保存结果
    os.makedirs("EHA_fig_FNO_optimized_with_kappa_d", exist_ok=True)
    system_final.visualize_results(results_final)

    # 保存最优参数到文件
    np.savetxt('best_parameters_fno_with_kappa_d.txt', best_pos,
               header=' '.join(param_names),
               fmt='%.6f')
    print("\n最优参数已保存到 'best_parameters_fno_with_kappa_d.txt'")

    # 保存优化配置
    with open('fno_optimization_config_with_kappa_d.txt', 'w') as f:
        f.write("FNO优化配置 (包含kappa_d)\n")
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

    print("优化配置已保存到 'fno_optimization_config_with_kappa_d.txt'")
    print("\nFNO优化完成！")


if __name__ == "__main__":
    main()