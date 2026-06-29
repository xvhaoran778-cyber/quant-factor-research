"""策略参数优化器 - 网格搜索 + 遗传算法"""

import itertools
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any
from loguru import logger
from dataclasses import dataclass


@dataclass
class OptimizeResult:
    """参数优化结果"""
    best_params: Dict[str, Any]
    best_return: float
    best_sharpe: float
    best_max_dd: float
    all_results: pd.DataFrame
    param_importance: Dict[str, float]


class GridSearchOptimizer:
    """网格搜索参数优化器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
    
    def optimize(self, strategy_class, param_grid: Dict[str, List],
                 df: pd.DataFrame, initial_capital: float = 1000000,
                 metric: str = 'sharpe') -> OptimizeResult:
        """网格搜索优化
        
        Args:
            strategy_class: 策略类
            param_grid: {参数名: [候选值列表]}
            df: K线数据
            metric: 优化目标 ('return', 'sharpe', 'calmar')
        """
        logger.info(f"开始网格搜索，参数组合数: {np.prod([len(v) for v in param_grid.values()])}")
        
        keys = list(param_grid.keys())
        results = []
        
        for values in itertools.product(*param_grid.values()):
            params = dict(zip(keys, values))
            
            try:
                strategy = strategy_class(params)
                result = strategy.backtest(df, initial_capital)
                
                equity = pd.Series(result['equity_curve'])
                daily_return = equity.pct_change().dropna()
                
                # 计算指标
                total_return = result['total_return']
                sharpe = np.sqrt(252) * daily_return.mean() / daily_return.std() if daily_return.std() > 0 else 0
                
                cummax = equity.cummax()
                drawdown = (equity - cummax) / cummax
                max_dd = drawdown.min() * 100
                calmar = -total_return / max_dd if max_dd != 0 else 0
                
                results.append({
                    **params,
                    'total_return': total_return,
                    'sharpe': sharpe,
                    'max_drawdown': max_dd,
                    'calmar': calmar,
                    'trades': result['total_trades']
                })
                
            except Exception as e:
                logger.warning(f"参数 {params} 回测失败: {e}")
        
        if not results:
            logger.error("所有参数组合回测失败")
            return None
        
        results_df = pd.DataFrame(results)
        
        # 按指标排序
        metric_map = {'return': 'total_return', 'sharpe': 'sharpe', 'calmar': 'calmar'}
        sort_col = metric_map.get(metric, 'sharpe')
        results_df = results_df.sort_values(sort_col, ascending=False)
        
        best = results_df.iloc[0]
        best_params = {k: best[k] for k in keys}
        
        logger.info(f"最优参数: {best_params}, {metric}={best[sort_col]:.4f}")
        
        return OptimizeResult(
            best_params={k: float(v) if isinstance(v, (np.floating, np.integer)) else int(v) if isinstance(v, np.integer) else v 
                        for k, v in best_params.items()},
            best_return=float(best['total_return']),
            best_sharpe=float(best['sharpe']),
            best_max_dd=float(best['max_drawdown']),
            all_results=results_df,
            param_importance={}
        )
    
    def _calc_param_importance(self, results_df: pd.DataFrame,
                               param_keys: List[str], metric_col: str) -> Dict[str, float]:
        """计算参数重要性（变异系数）"""
        importance = {}
        for key in param_keys:
            groups = results_df.groupby(key)[metric_col].agg(['mean', 'std'])
            cv = (groups['std'] / (groups['mean'].abs() + 1e-8)).mean()
            importance[key] = float(cv)
        return importance


class GeneticOptimizer:
    """遗传算法参数优化器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.population_size = self.config.get('population_size', 30)
        self.generations = self.config.get('generations', 20)
        self.mutation_rate = self.config.get('mutation_rate', 0.2)
        self.crossover_rate = self.config.get('crossover_rate', 0.7)
        self.elite_count = self.config.get('elite_count', 5)
    
    def optimize(self, strategy_class, param_bounds: Dict[str, Tuple],
                 df: pd.DataFrame, initial_capital: float = 1000000,
                 metric: str = 'sharpe') -> OptimizeResult:
        """遗传算法优化
        
        Args:
            strategy_class: 策略类
            param_bounds: {参数名: (最小值, 最大值, 步长)}，支持int和float
        """
        logger.info(f"开始遗传算法优化，种群: {self.population_size}，代数: {self.generations}")
        
        param_configs = []
        for name, bounds in param_bounds.items():
            if len(bounds) == 3:
                param_configs.append({
                    'name': name,
                    'min': bounds[0],
                    'max': bounds[1],
                    'step': bounds[2],
                    'is_int': isinstance(bounds[2], int) or bounds[2] == int(bounds[2])
                })
            else:
                param_configs.append({
                    'name': name,
                    'min': bounds[0],
                    'max': bounds[1],
                    'step': None,
                    'is_int': False
                })
        
        # 初始化种群
        population = self._initialize_population(param_configs)
        all_results = []
        
        for gen in range(self.generations):
            # 评估种群
            fitness = []
            for individual in population:
                result = self._evaluate(strategy_class, individual, df, initial_capital, metric)
                if result is not None:
                    fitness.append((individual, result))
            
            if not fitness:
                continue
            
            fitness.sort(key=lambda x: x[1]['score'], reverse=True)
            best = fitness[0]
            
            logger.info(f"Gen {gen+1}/{self.generations}: best {metric}={best[1]['score']:.4f}, "
                       f"params={best[0]}")
            
            # 保存结果
            for individual, result in fitness:
                all_results.append({**individual, **result})
            
            # 选择精英
            elite = [ind for ind, _ in fitness[:self.elite_count]]
            
            # 生成新一代
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate and len(fitness) >= 2:
                    # 交叉
                    p1 = random.choice(fitness[:10])[0]
                    p2 = random.choice(fitness[:10])[0]
                    child = self._crossover(p1, p2, param_configs)
                    new_population.append(child)
                elif random.random() < self.mutation_rate:
                    # 变异
                    parent = random.choice(fitness[:5])[0]
                    child = self._mutate(parent, param_configs)
                    new_population.append(child)
                else:
                    new_population.append(random.choice(fitness[:10])[0])
            
            population = new_population
        
        # 汇总结果
        results_df = pd.DataFrame(all_results)
        metric_map = {'return': 'total_return', 'sharpe': 'sharpe', 'calmar': 'calmar'}
        sort_col = metric_map.get(metric, 'sharpe')
        results_df = results_df.sort_values('score', ascending=False)
        
        best = results_df.iloc[0]
        best_params = {c['name']: best[c['name']] for c in param_configs}
        
        return OptimizeResult(
            best_params={k: float(v) if isinstance(v, (np.floating, np.integer)) else int(v) if isinstance(v, np.integer) else v 
                        for k, v in best_params.items()},
            best_return=float(best['total_return']),
            best_sharpe=float(best['sharpe']),
            best_max_dd=float(best['max_drawdown']),
            all_results=results_df,
            param_importance={k: float(v) for k, v in importance.items()}
        )
    
    def _initialize_population(self, param_configs: List[Dict]) -> List[Dict]:
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for cfg in param_configs:
                if cfg['is_int'] or cfg.get('step') and isinstance(cfg['step'], int):
                    val = random.randint(int(cfg['min']), int(cfg['max']))
                elif cfg.get('step'):
                    steps = int((cfg['max'] - cfg['min']) / cfg['step'])
                    val = cfg['min'] + random.randint(0, steps) * cfg['step']
                else:
                    val = random.uniform(cfg['min'], cfg['max'])
                individual[cfg['name']] = val
            population.append(individual)
        return population
    
    def _evaluate(self, strategy_class, params: Dict, df: pd.DataFrame,
                  capital: float, metric: str) -> Dict:
        """评估个体"""
        try:
            strategy = strategy_class(params)
            result = strategy.backtest(df, capital)
            
            equity = pd.Series(result['equity_curve'])
            daily_return = equity.pct_change().dropna()
            
            total_return = result['total_return']
            sharpe = np.sqrt(252) * daily_return.mean() / daily_return.std() if daily_return.std() > 0 else 0
            
            cummax = equity.cummax()
            drawdown = (equity - cummax) / cummax
            max_dd = drawdown.min() * 100
            calmar = -total_return / max_dd if max_dd != 0 else 0
            
            score_map = {'return': total_return, 'sharpe': sharpe, 'calmar': calmar}
            score = score_map.get(metric, sharpe)
            
            return {
                'total_return': total_return,
                'sharpe': sharpe,
                'calmar': calmar,
                'max_drawdown': max_dd,
                'trades': result['total_trades'],
                'score': score
            }
        except:
            return None
    
    def _crossover(self, p1: Dict, p2: Dict, param_configs: List[Dict]) -> Dict:
        """均匀交叉"""
        child = {}
        for cfg in param_configs:
            child[cfg['name']] = p1[cfg['name']] if random.random() < 0.5 else p2[cfg['name']]
        return child
    
    def _mutate(self, parent: Dict, param_configs: List[Dict]) -> Dict:
        """高斯变异"""
        child = parent.copy()
        cfg = random.choice(param_configs)
        if cfg['is_int'] or cfg.get('step') and isinstance(cfg['step'], int):
            delta = random.randint(-5, 5)
            child[cfg['name']] = max(int(cfg['min']), min(int(cfg['max']), int(child[cfg['name']]) + delta))
        else:
            delta = random.gauss(0, (cfg['max'] - cfg['min']) * 0.1)
            child[cfg['name']] = max(cfg['min'], min(cfg['max'], child[cfg['name']] + delta))
        return child
