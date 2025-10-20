"""
权重优化算法集合
包含网格搜索、贝叶斯优化、强化学习等多种优化方法
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import random
from abc import ABC, abstractmethod
import json
from pathlib import Path

# 可选的高级优化库
try:
    from skopt import gp_minimize
    from skopt.space import Real
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

@dataclass
class WeightOptimizationConfig:
    """权重优化配置"""
    # 基础配置
    min_dense_weight: float = 0.1
    max_dense_weight: float = 0.9
    weight_step: float = 0.05  # 网格搜索步长

    # 贝叶斯优化配置
    bayesian_n_calls: int = 30
    bayesian_n_initial_points: int = 10

    # 强化学习配置
    rl_learning_rate: float = 0.001
    rl_epsilon: float = 0.1
    rl_gamma: float = 0.95
    rl_memory_size: int = 1000
    rl_batch_size: int = 32

    # 早停配置
    patience: int = 10
    min_improvement: float = 0.001

class WeightOptimizer(ABC):
    """权重优化器基类"""

    def __init__(self, config: WeightOptimizationConfig):
        self.config = config
        self.optimization_history = []

    @abstractmethod
    def optimize(self, evaluation_function: Callable, validation_data: List[Dict]) -> Tuple[float, float, Dict]:
        """优化权重"""
        pass

    def evaluate_weights(self, dense_weight: float, sparse_weight: float,
                        evaluation_function: Callable, validation_data: List[Dict]) -> float:
        """评估给定权重的性能"""
        try:
            # 确保权重和为1
            total_weight = dense_weight + sparse_weight
            if total_weight <= 0:
                return float('-inf')

            normalized_dense = dense_weight / total_weight
            normalized_sparse = sparse_weight / total_weight

            # 调用评估函数
            score = evaluation_function(normalized_dense, normalized_sparse, validation_data)

            # 记录历史
            self.optimization_history.append({
                'dense_weight': normalized_dense,
                'sparse_weight': normalized_sparse,
                'score': score,
                'timestamp': np.datetime64('now')
            })

            return score

        except Exception as e:
            print(f"权重评估失败: {e}")
            return float('-inf')

class GridSearchOptimizer(WeightOptimizer):
    """网格搜索权重优化器"""

    def optimize(self, evaluation_function: Callable, validation_data: List[Dict]) -> Tuple[float, float, Dict]:
        """网格搜索优化权重"""
        print("🔍 开始网格搜索权重优化...")

        best_score = float('-inf')
        best_weights = (0.7, 0.3)  # 默认值

        # 生成搜索网格
        dense_weights = np.arange(
            self.config.min_dense_weight,
            self.config.max_dense_weight + self.config.weight_step,
            self.config.weight_step
        )

        total_combinations = len(dense_weights)
        print(f"📊 搜索空间: {total_combinations} 种组合")

        for i, dense_w in enumerate(dense_weights):
            sparse_w = 1.0 - dense_w

            # 评估当前权重
            score = self.evaluate_weights(dense_w, sparse_w, evaluation_function, validation_data)

            if score > best_score:
                best_score = score
                best_weights = (dense_w, sparse_w)
                print(f"✅ 找到更好权重: dense={dense_w:.3f}, sparse={sparse_w:.3f}, score={score:.4f}")

            # 进度显示
            if (i + 1) % 5 == 0:
                progress = (i + 1) / total_combinations * 100
                print(f"⏱️  进度: {progress:.1f}% ({i+1}/{total_combinations})")

        print(f"🎯 网格搜索完成! 最佳权重: dense={best_weights[0]:.3f}, sparse={best_weights[1]:.3f}")

        return best_weights[0], best_weights[1], {
            'method': 'grid_search',
            'best_score': best_score,
            'total_evaluations': len(dense_weights),
            'optimization_history': self.optimization_history
        }

class BayesianOptimizer(WeightOptimizer):
    """贝叶斯优化权重优化器"""

    def __init__(self, config: WeightOptimizationConfig):
        super().__init__(config)
        if not BAYESIAN_AVAILABLE:
            raise ImportError("需要安装scikit-optimize: pip install scikit-optimize")

    def optimize(self, evaluation_function: Callable, validation_data: List[Dict]) -> Tuple[float, float, Dict]:
        """贝叶斯优化权重"""
        print("🧠 开始贝叶斯优化权重...")

        def objective(weights):
            dense_w, sparse_w = weights
            # 确保权重和为1
            if abs(dense_w + sparse_w - 1.0) > 1e-6:
                return 1e6  # 大惩罚

            # 边界检查
            if not (self.config.min_dense_weight <= dense_w <= self.config.max_dense_weight):
                return 1e6
            if not (self.config.min_dense_weight <= sparse_w <= self.config.max_dense_weight):
                return 1e6

            # 评估权重
            score = self.evaluate_weights(dense_w, sparse_w, evaluation_function, validation_data)
            return -score  # 贝叶斯优化是最小化问题

        # 定义搜索空间
        search_space = [
            Real(self.config.min_dense_weight, self.config.max_dense_weight, name='dense_weight'),
            Real(self.config.min_dense_weight, self.config.max_dense_weight, name='sparse_weight')
        ]

        # 执行贝叶斯优化
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=self.config.bayesian_n_calls,
            n_initial_points=self.config.bayesian_n_initial_points,
            random_state=42
        )

        best_dense_weight = result.x[0]
        best_sparse_weight = result.x[1]
        best_score = -result.fun

        print(f"🎯 贝叶斯优化完成! 最佳权重: dense={best_dense_weight:.4f}, sparse={best_sparse_weight:.4f}")
        print(f"📈 最佳得分: {best_score:.4f}")

        return best_dense_weight, best_sparse_weight, {
            'method': 'bayesian_optimization',
            'best_score': best_score,
            'total_evaluations': len(result.func_vals),
            'convergence_history': result.func_vals.tolist(),
            'optimization_history': self.optimization_history
        }

class ReinforcementLearningOptimizer(WeightOptimizer):
    """强化学习权重优化器"""

    def __init__(self, config: WeightOptimizationConfig):
        super().__init__(config)
        if not RL_AVAILABLE:
            raise ImportError("需要安装PyTorch: pip install torch")

        self.state_size = 10  # 状态特征维度
        self.action_size = 5  # 离散化动作空间
        self.q_network = self._build_q_network()
        self.memory = []  # 经验回放缓冲区
        self.epsilon = config.rl_epsilon

    def _build_q_network(self) -> nn.Module:
        """构建Q网络"""
        class QNetwork(nn.Module):
            def __init__(self, state_size, action_size):
                super(QNetwork, self).__init__()
                self.fc1 = nn.Linear(state_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, action_size)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.fc3(x)

        return QNetwork(self.state_size, self.action_size)

    def get_state_features(self, validation_data: List[Dict]) -> np.ndarray:
        """提取状态特征"""
        if not validation_data:
            return np.zeros(self.state_size)

        # 1. 查询复杂度特征
        avg_question_length = np.mean([len(item.get('question', '').split()) for item in validation_data])

        # 2. 领域多样性特征
        domains = [item.get('domain', 'general') for item in validation_data]
        domain_diversity = len(set(domains)) / len(domains) if domains else 0

        # 3. 分数分布特征
        all_dense_scores = []
        all_sparse_scores = []

        for item in validation_data:
            if 'dense_scores' in item:
                all_dense_scores.extend(item['dense_scores'])
            if 'sparse_scores' in item:
                all_sparse_scores.extend(item['sparse_scores'])

        avg_dense = np.mean(all_dense_scores) if all_dense_scores else 0.5
        avg_sparse = np.mean(all_sparse_scores) if all_sparse_scores else 0.5
        std_dense = np.std(all_dense_scores) if all_dense_scores else 0.1
        std_sparse = np.std(all_sparse_scores) if all_sparse_scores else 0.1

        # 4. 样本数量特征
        n_samples = len(validation_data)

        # 5. 性能特征（需要历史数据）
        recent_performance = 0.5  # 默认值
        if self.optimization_history:
            recent_scores = [h['score'] for h in self.optimization_history[-5:]]
            recent_performance = np.mean(recent_scores) if recent_scores else 0.5

        # 组合特征向量
        features = np.array([
            avg_question_length / 50.0,  # 归一化
            domain_diversity,
            avg_dense,
            avg_sparse,
            std_dense,
            std_sparse,
            min(n_samples / 100.0, 1.0),  # 归一化
            recent_performance,
            abs(avg_dense - avg_sparse),  # 分数差异
            len(set(domains)) / 10.0  # 领域数量归一化
        ])

        return np.clip(features, 0, 1)  # 确保特征在[0,1]范围内

    def select_action(self, state: np.ndarray) -> int:
        """选择动作（权重调整）"""
        if random.random() < self.epsilon:
            # 探索：随机选择
            return random.randint(0, self.action_size - 1)
        else:
            # 利用：选择Q值最大的动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def action_to_weights(self, action: int) -> Tuple[float, float]:
        """将动作转换为权重"""
        # 将离散动作映射到连续权重空间
        weight_adjustments = np.linspace(-0.2, 0.2, self.action_size)
        adjustment = weight_adjustments[action]

        base_dense = 0.7
        base_sparse = 0.3

        adjusted_dense = base_dense + adjustment
        adjusted_sparse = base_sparse - adjustment

        # 确保在合理范围内
        adjusted_dense = max(self.config.min_dense_weight, min(self.config.max_dense_weight, adjusted_dense))
        adjusted_sparse = max(self.config.min_dense_weight, min(self.config.max_dense_weight, adjusted_sparse))

        # 归一化
        total = adjusted_dense + adjusted_sparse
        return adjusted_dense / total, adjusted_sparse / total

    def store_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """存储经验"""
        experience = (state, action, reward, next_state)
        self.memory.append(experience)

        # 限制内存大小
        if len(self.memory) > self.config.rl_memory_size:
            self.memory.pop(0)

    def train_q_network(self):
        """训练Q网络"""
        if len(self.memory) < self.config.rl_batch_size:
            return

        # 随机采样批次
        batch = random.sample(self.memory, self.config.rl_batch_size)

        # 准备训练数据
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor([exp[3] for exp in batch])

        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]
            target_q_values = rewards + self.config.rl_gamma * next_q_values

        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # 反向传播
        optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.config.rl_learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def optimize(self, evaluation_function: Callable, validation_data: List[Dict]) -> Tuple[float, float, Dict]:
        """强化学习优化权重"""
        print("🤖 开始强化学习权重优化...")

        episode_rewards = []
        best_weights = (0.7, 0.3)
        best_score = float('-inf')

        # 获取初始状态
        current_state = self.get_state_features(validation_data)

        for episode in range(self.config.bayesian_n_calls):  # 复用参数
            # 选择动作
            action = self.select_action(current_state)
            dense_weight, sparse_weight = self.action_to_weights(action)

            # 执行动作并获取奖励
            score = self.evaluate_weights(dense_weight, sparse_weight, evaluation_function, validation_data)
            reward = score  # 奖励就是性能得分

            # 获取下一个状态（这里简化，实际应该基于动作结果更新状态）
            next_state = self.get_state_features(validation_data)

            # 存储经验
            self.store_experience(current_state, action, reward, next_state)

            # 训练Q网络
            self.train_q_network()

            # 更新最佳权重
            if score > best_score:
                best_score = score
                best_weights = (dense_weight, sparse_weight)

            # 更新状态
            current_state = next_state

            # 记录奖励
            episode_rewards.append(reward)

            # 进度显示
            if (episode + 1) % 5 == 0:
                avg_reward = np.mean(episode_rewards[-5:])
                print(f"Episode {episode+1}: avg_reward={avg_reward:.4f}, epsilon={self.epsilon:.3f}")

            # 衰减探索率
            self.epsilon *= 0.995

        print(f"🎯 强化学习优化完成! 最佳权重: dense={best_weights[0]:.4f}, sparse={best_weights[1]:.4f}")

        return best_weights[0], best_weights[1], {
            'method': 'reinforcement_learning',
            'best_score': best_score,
            'episode_rewards': episode_rewards,
            'final_epsilon': self.epsilon,
            'optimization_history': self.optimization_history
        }

class WeightOptimizationManager:
    """权重优化管理器"""

    def __init__(self, config: WeightOptimizationConfig):
        self.config = config
        self.optimizers = {
            'grid_search': GridSearchOptimizer(config),
            'bayesian': BayesianOptimizer(config) if BAYESIAN_AVAILABLE else None,
            'reinforcement_learning': ReinforcementLearningOptimizer(config) if RL_AVAILABLE else None
        }

    def optimize_weights(self, method: str, evaluation_function: Callable,
                        validation_data: List[Dict]) -> Tuple[float, float, Dict]:
        """使用指定方法优化权重"""

        if method not in self.optimizers or self.optimizers[method] is None:
            available_methods = [k for k, v in self.optimizers.items() if v is not None]
            raise ValueError(f"优化方法 {method} 不可用。可用方法: {available_methods}")

        optimizer = self.optimizers[method]
        print(f"🔧 使用 {method} 方法进行权重优化...")

        return optimizer.optimize(evaluation_function, validation_data)

    def compare_optimization_methods(self, evaluation_function: Callable,
                                   validation_data: List[Dict],
                                   methods: Optional[List[str]] = None) -> Dict:
        """比较不同的优化方法"""

        if methods is None:
            methods = [k for k, v in self.optimizers.items() if v is not None]

        results = {}

        for method in methods:
            try:
                dense_weight, sparse_weight, optimization_info = self.optimize_weights(
                    method, evaluation_function, validation_data
                )

                results[method] = {
                    'dense_weight': dense_weight,
                    'sparse_weight': sparse_weight,
                    'best_score': optimization_info['best_score'],
                    'optimization_info': optimization_info
                }

            except Exception as e:
                results[method] = {
                    'error': str(e),
                    'status': 'failed'
                }

        # 找出最佳方法
        best_method = None
        best_score = float('-inf')

        for method, result in results.items():
            if 'best_score' in result and result['best_score'] > best_score:
                best_score = result['best_score']
                best_method = method

        results['best_method'] = best_method
        results['best_weights'] = results[best_method] if best_method else None

        return results

# 便捷函数
def optimize_weights_simple(validation_data: List[Dict], evaluation_function: Callable,
                          method: str = 'grid_search') -> Tuple[float, float, Dict]:
    """简单权重优化"""
    config = WeightOptimizationConfig()
    manager = WeightOptimizationManager(config)
    return manager.optimize_weights(method, evaluation_function, validation_data)

def compare_optimization_methods(validation_data: List[Dict], evaluation_function: Callable) -> Dict:
    """比较所有可用的优化方法"""
    config = WeightOptimizationConfig()
    manager = WeightOptimizationManager(config)
    return manager.compare_optimization_methods(evaluation_function, validation_data)