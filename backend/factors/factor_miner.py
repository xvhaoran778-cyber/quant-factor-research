"""RD-Agent因子挖掘 - 自动挖掘和优化因子"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Tuple
from loguru import logger
import random
from dataclasses import dataclass


@dataclass
class Factor:
    """因子定义"""
    name: str
    expression: str  # 因子表达式
    category: str  # 因子类别
    ic: float = 0  # IC值
    ir: float = 0  # IR值
    turnover: float = 0  # 换手率
    score: float = 0  # 综合评分


class FactorMiner:
    """因子挖掘器 - 参考RD-Agent的自动因子挖掘"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.population_size = self.config.get('population_size', 50)
        self.generations = self.config.get('generations', 20)
        self.mutation_rate = self.config.get('mutation_rate', 0.3)
        self.crossover_rate = self.config.get('crossover_rate', 0.7)
        
        # 基础算子
        self.operators = {
            'unary': ['abs', 'log', 'sqrt', 'rank', 'zscore'],
            'binary': ['add', 'sub', 'mul', 'div', 'corr', 'cov'],
            'rolling': ['mean', 'std', 'max', 'min', 'skew', 'kurt']
        }
        
        # 基础特征
        self.base_features = ['open', 'high', 'low', 'close', 'volume']
    
    def mine_factors(self, df: pd.DataFrame, target_col: str = 'return_1d',
                    n_factors: int = 10) -> List[Factor]:
        """挖掘因子"""
        logger.info("开始因子挖掘...")
        
        # 计算目标变量
        df = df.copy()
        df['return_1d'] = df['close'].pct_change().shift(-1)
        
        # 初始化种群
        population = self._initialize_population()
        
        best_factors = []
        
        for gen in range(self.generations):
            # 评估种群
            fitness_scores = []
            for individual in population:
                try:
                    factor_values = self._evaluate_expression(df, individual)
                    if factor_values is not None:
                        ic = self._calculate_ic(factor_values, df[target_col])
                        fitness_scores.append((individual, ic))
                    else:
                        fitness_scores.append((individual, 0))
                except:
                    fitness_scores.append((individual, 0))
            
            # 排序
            fitness_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # 保存最佳因子
            if fitness_scores and abs(fitness_scores[0][1]) > 0.03:
                best_expr, best_ic = fitness_scores[0]
                factor = Factor(
                    name=f"mined_factor_{gen}",
                    expression=best_expr,
                    category="mined",
                    ic=best_ic,
                    score=abs(best_ic) * 100
                )
                best_factors.append(factor)
            
            # 选择、交叉、变异
            population = self._evolve_population(fitness_scores)
            
            if (gen + 1) % 5 == 0:
                logger.info(f"Generation {gen+1}/{self.generations}, Best IC: {fitness_scores[0][1]:.4f}")
        
        # 去重和排序
        best_factors = self._deduplicate_factors(best_factors)
        best_factors.sort(key=lambda x: abs(x.ic), reverse=True)
        
        return best_factors[:n_factors]
    
    def _initialize_population(self) -> List[str]:
        """初始化种群"""
        population = []
        
        for _ in range(self.population_size):
            # 随机生成因子表达式
            expr = self._random_expression()
            population.append(expr)
        
        return population
    
    def _random_expression(self, depth: int = 0) -> str:
        """随机生成表达式"""
        if depth > 3:
            return random.choice(self.base_features)
        
        r = random.random()
        
        if r < 0.3:
            # 一元运算
            op = random.choice(self.operators['unary'])
            expr = self._random_expression(depth + 1)
            return f"{op}({expr})"
        elif r < 0.6:
            # 二元运算
            op = random.choice(self.operators['binary'])
            expr1 = self._random_expression(depth + 1)
            expr2 = self._random_expression(depth + 1)
            if op in ['add', 'sub', 'mul', 'div']:
                return f"({expr1} {op[0]} {expr2})"
            else:
                return f"{op}({expr1}, {expr2})"
        elif r < 0.8:
            # 滚动运算
            op = random.choice(self.operators['rolling'])
            expr = self._random_expression(depth + 1)
            window = random.choice([5, 10, 20])
            return f"rolling_{op}({expr}, {window})"
        else:
            return random.choice(self.base_features)
    
    def _evaluate_expression(self, df: pd.DataFrame, expr: str) -> pd.Series:
        """计算表达式"""
        try:
            # 简化实现：使用eval计算表达式
            # 实际应该使用更安全的表达式解析器
            
            # 替换基础特征
            eval_expr = expr
            for feat in self.base_features:
                if feat in df.columns:
                    eval_expr = eval_expr.replace(feat, f"df['{feat}']")
            
            # 处理一元运算
            eval_expr = eval_expr.replace('abs(', 'np.abs(')
            eval_expr = eval_expr.replace('log(', 'np.log(np.abs(') 
            eval_expr = eval_expr.replace('sqrt(', 'np.sqrt(np.abs(')
            
            # 处理滚动运算
            import re
            rolling_pattern = r'rolling_(\w+)\(([^,]+),\s*(\d+)\)'
            def replace_rolling(match):
                op = match.group(1)
                col = match.group(2)
                window = match.group(3)
                return f"{col}.rolling({window}).{op}()"
            eval_expr = re.sub(rolling_pattern, replace_rolling, eval_expr)
            
            result = eval(eval_expr)
            
            if isinstance(result, pd.Series):
                return result.replace([np.inf, -np.inf], np.nan)
            return None
        except:
            return None
    
    def _calculate_ic(self, factor_values: pd.Series, returns: pd.Series) -> float:
        """计算IC值"""
        try:
            valid_idx = factor_values.notna() & returns.notna()
            if valid_idx.sum() < 30:
                return 0
            
            from scipy import stats
            ic, _ = stats.spearmanr(factor_values[valid_idx], returns[valid_idx])
            return ic if not np.isnan(ic) else 0
        except:
            return 0
    
    def _evolve_population(self, fitness_scores: List[Tuple[str, float]]) -> List[str]:
        """进化种群"""
        # 选择前50%
        selected = [x[0] for x in fitness_scores[:len(fitness_scores)//2]]
        
        new_population = []
        
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(selected) >= 2:
                # 交叉
                parent1, parent2 = random.sample(selected, 2)
                child = self._crossover(parent1, parent2)
                new_population.append(child)
            elif random.random() < self.mutation_rate:
                # 变异
                parent = random.choice(selected)
                child = self._mutate(parent)
                new_population.append(child)
            else:
                # 复制
                new_population.append(random.choice(selected))
        
        return new_population
    
    def _crossover(self, parent1: str, parent2: str) -> str:
        """交叉"""
        # 简单实现：随机选择一个
        return random.choice([parent1, parent2])
    
    def _mutate(self, parent: str) -> str:
        """变异"""
        # 简单实现：重新生成
        return self._random_expression()
    
    def _deduplicate_factors(self, factors: List[Factor]) -> List[Factor]:
        """去重"""
        seen = set()
        unique_factors = []
        for factor in factors:
            if factor.expression not in seen:
                seen.add(factor.expression)
                unique_factors.append(factor)
        return unique_factors


class FactorOptimizer:
    """因子优化器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def optimize_factors(self, df: pd.DataFrame, factors: List[Factor],
                        target_col: str = 'return_1d') -> List[Factor]:
        """优化因子组合"""
        logger.info("开始因子优化...")
        
        # 计算因子相关性矩阵
        factor_values = {}
        for factor in factors:
            miner = FactorMiner()
            values = miner._evaluate_expression(df, factor.expression)
            if values is not None:
                factor_values[factor.name] = values
        
        if not factor_values:
            return factors
        
        factor_df = pd.DataFrame(factor_values)
        corr_matrix = factor_df.corr()
        
        # 选择低相关性的因子
        selected_factors = []
        for factor in factors:
            if factor.name in corr_matrix.columns:
                # 检查与已选因子的相关性
                is_redundant = False
                for selected in selected_factors:
                    if selected.name in corr_matrix.columns:
                        corr = abs(corr_matrix.loc[factor.name, selected.name])
                        if corr > 0.8:
                            is_redundant = True
                            break
                
                if not is_redundant:
                    selected_factors.append(factor)
        
        logger.info(f"优化完成，保留{len(selected_factors)}个低相关性因子")
        return selected_factors
    
    def calculate_composite_score(self, factors: List[Factor]) -> pd.Series:
        """计算综合因子得分"""
        # 基于IC加权
        weights = [abs(f.ic) for f in factors]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return pd.Series(0, index=range(len(factors)))
        
        weights = [w/total_weight for w in weights]
        
        # 加权平均
        composite = pd.Series(0, index=range(len(factors)))
        for factor, weight in zip(factors, weights):
            composite += factor.score * weight
        
        return composite
