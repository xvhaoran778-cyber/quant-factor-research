"""
因子对比实验框架

对比 6 个实验：
1. Baseline: Alpha191 (α132 + ret60 + size)
2. Baseline + Agent
3. New Factors: 15 个新因子
4. New Factors + Agent
5. Hybrid: Alpha191 + 新因子
6. Hybrid + Agent
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from app.services.new_factors import FACTOR_REGISTRY, compute_all_factors
from app.services.factor_ic_analysis import analyze_factor, batch_analyze_factors
from app.services.research_backtest import run_scored_backtest, build_weekly_feature_panel
from app.services.strategies.v8_canonical import v8_scorer, v8_candidate_filter, merge_alpha_132
from app.services.strategies.agent_gate import agent_gate


class ComparisonExperiment:
    """对比实验类"""
    
    def __init__(self, data_dir: str, start_date: str, end_date: str):
        """
        初始化实验
        
        Args:
            data_dir: 数据目录
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        self.data_dir = data_dir
        self.start_date = pd.to_datetime(start_date).date()
        self.end_date = pd.to_datetime(end_date).date()
        
        self.results = {}
        self.factor_data = {}
        
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        
        # 初始化市场数据存储
        from app.services.market_store import ParquetMarketStore
        store = ParquetMarketStore(self.data_dir)
        
        # 构建周频面板
        self.panel = build_weekly_feature_panel(
            store,
            self.start_date,
            self.end_date
        )
        
        # 合并 α132 因子
        self.panel_with_alpha132 = merge_alpha_132(self.panel)
        
        print(f"数据加载完成: {len(self.panel)} 行")
        
    def prepare_new_factors(self):
        """准备新因子"""
        print("\n准备新因子...")
        
        # 从市场数据存储加载原始日线数据
        from app.services.market_store import ParquetMarketStore
        store = ParquetMarketStore(self.data_dir)
        
        # 使用 read 方法一次性加载所有股票的日线数据
        print(f"加载 {self.start_date} 至 {self.end_date} 的日线数据...")
        all_daily = store.read(self.start_date, self.end_date, symbols=None, fill_suspensions=False)
        
        if all_daily.empty:
            raise ValueError("无法加载任何股票的日线数据")
        
        print(f"加载了 {len(all_daily)} 条日线数据，覆盖 {all_daily['symbol'].nunique()} 只股票")
        
        # 构建透视表
        close_prices = all_daily.pivot(index='date', columns='symbol', values='close')
        open_prices = all_daily.pivot(index='date', columns='symbol', values='open')
        high_prices = all_daily.pivot(index='date', columns='symbol', values='high')
        low_prices = all_daily.pivot(index='date', columns='symbol', values='low')
        volume = all_daily.pivot(index='date', columns='symbol', values='volume')
        
        # 计算收益率
        returns = close_prices.pct_change()
        
        # 计算市场收益率
        market_returns = returns.mean(axis=1)
        
        # 计算 250 日最高价
        high_250d = close_prices.rolling(250).max()
        
        # 计算前一日收盘价
        prev_close = close_prices.shift(1)
        
        # 生成行业收益率（简单模拟：将股票分成3个行业）
        symbols = close_prices.columns.tolist()
        n_industries = 3
        industry_returns = pd.DataFrame(index=close_prices.index)
        for i in range(n_industries):
            industry_col = symbols[i::n_industries]
            industry_returns[f'INDUSTRY_{i}'] = returns[industry_col].mean(axis=1)
        
        # 准备数据字典
        self.factor_input_data = {
            'close_prices_matrix': close_prices,
            'stock_returns': returns,
            'market_returns': market_returns,
            'industry_returns': industry_returns,
            'close': close_prices,
            'open': open_prices,
            'open_price': open_prices,
            'high': high_prices,
            'low': low_prices,
            'volume': volume,
            'returns': returns,
            'prices': close_prices,
            'high_250d': high_250d,
            'prev_close': prev_close
        }
        
        print("新因子数据准备完成")
        
    def compute_new_factors(self, factor_names: List[str] = None):
        """
        计算新因子
        
        Args:
            factor_names: 要计算的因子名称列表，None 表示全部
        """
        if factor_names is None:
            factor_names = list(FACTOR_REGISTRY.keys())
        
        print(f"\n计算 {len(factor_names)} 个新因子...")
        
        # 计算因子
        factor_df = compute_all_factors(self.factor_input_data, factor_names)
        
        # factor_df 的索引是 (date, symbol) 的 MultiIndex
        # 重置索引，将 MultiIndex 转换为普通列
        factor_df_reset = factor_df.reset_index()
        
        # 确保 date 列是 datetime 类型
        factor_df_reset['date'] = pd.to_datetime(factor_df_reset['date'])
        
        # 合并到面板
        self.panel = self.panel.merge(
            factor_df_reset,
            on=['date', 'symbol'],
            how='left'
        )
        
        print(f"新因子计算完成，面板现在有 {len(self.panel.columns)} 列")
        
    def analyze_new_factors_ic(self, forward_days: int = 5):
        """
        分析新因子的 IC
        
        Args:
            forward_days: 前瞻天数
        """
        print(f"\n分析新因子 IC (前瞻 {forward_days} 天)...")
        
        # 计算前瞻收益率
        close_prices = self.panel.pivot(index='date', columns='symbol', values='close')
        forward_returns = close_prices.pct_change(forward_days).shift(-forward_days)
        
        # 为每个新因子计算 IC
        factor_dict = {}
        for factor_name in FACTOR_REGISTRY.keys():
            if factor_name in self.panel.columns:
                factor_values = self.panel.pivot(
                    index='date', 
                    columns='symbol', 
                    values=factor_name
                )
                factor_dict[factor_name] = factor_values
        
        # 批量分析
        self.ic_analysis_results = batch_analyze_factors(
            factor_dict, 
            forward_returns,
            n_groups=5
        )
        
        return self.ic_analysis_results
        
    def select_best_factors(self, top_n: int = 5) -> List[str]:
        """
        选择最佳因子
        
        Args:
            top_n: 选择的因子数量
        
        Returns:
            最佳因子名称列表
        """
        if 'ic_analysis_results' not in dir(self):
            raise ValueError("请先运行 analyze_new_factors_ic()")
        
        # 按 IC_IR 排序，选择 top_n
        best_factors = self.ic_analysis_results.nlargest(top_n, 'ic_ir')['factor_name'].tolist()
        
        print(f"\n选择 {top_n} 个最佳因子:")
        for i, factor in enumerate(best_factors, 1):
            ic_ir = self.ic_analysis_results[
                self.ic_analysis_results['factor_name'] == factor
            ]['ic_ir'].values[0]
            print(f"{i}. {factor} (IC_IR: {ic_ir:.4f})")
        
        return best_factors
    
    def create_new_factor_scorer(self, factor_names: List[str], 
                                 weights: List[float] = None):
        """
        创建新因子评分函数
        
        Args:
            factor_names: 因子名称列表
            weights: 因子权重列表，None 表示等权
        """
        if weights is None:
            weights = [1.0 / len(factor_names)] * len(factor_names)
        
        # 获取因子方向
        directions = []
        for factor_name in factor_names:
            if factor_name in FACTOR_REGISTRY:
                direction = FACTOR_REGISTRY[factor_name]['direction']
                directions.append(1 if direction == 'positive' else -1)
            else:
                directions.append(1)
        
        def new_factor_scorer(panel_with_factors: pd.DataFrame) -> pd.Series:
            """新因子评分函数"""
            scores = pd.Series(0.0, index=panel_with_factors.index)
            
            for factor_name, weight, direction in zip(factor_names, weights, directions):
                if factor_name in panel_with_factors.columns:
                    factor_values = panel_with_factors[factor_name].copy()
                    
                    # 处理 NaN
                    factor_values = factor_values.fillna(factor_values.median())
                    
                    # 排名标准化
                    factor_rank = factor_values.rank(pct=True)
                    
                    # 根据方向调整
                    factor_rank = factor_rank * direction
                    
                    # 加权
                    scores += factor_rank * weight
            
            return scores
        
        self.new_factor_scorer = new_factor_scorer
        self.new_factor_names = factor_names
        self.new_factor_weights = weights
        
        print(f"\n创建新因子评分函数:")
        print(f"因子: {factor_names}")
        print(f"权重: {weights}")
        print(f"方向: {directions}")
        
    def create_hybrid_scorer(self, alpha132_weight: float = 0.3,
                            new_factor_weight: float = 0.7):
        """
        创建混合评分函数（Alpha191 + 新因子）
        
        Args:
            alpha132_weight: Alpha132 权重
            new_factor_weight: 新因子权重
        """
        if not hasattr(self, 'new_factor_scorer'):
            raise ValueError("请先创建新因子评分函数")
        
        def hybrid_scorer(panel_with_factors: pd.DataFrame) -> pd.Series:
            """混合评分函数"""
            # Alpha132 评分
            alpha132_score = v8_scorer(panel_with_factors)
            alpha132_score = alpha132_score.rank(pct=True)
            
            # 新因子评分
            new_score = self.new_factor_scorer(panel_with_factors)
            new_score = new_score.rank(pct=True)
            
            # 混合
            hybrid_score = (
                alpha132_score * alpha132_weight +
                new_score * new_factor_weight
            )
            
            return hybrid_score
        
        self.hybrid_scorer = hybrid_scorer
        
        print(f"\n创建混合评分函数:")
        print(f"Alpha132 权重: {alpha132_weight}")
        print(f"新因子权重: {new_factor_weight}")
    
    def run_experiment(self, experiment_name: str, scorer, 
                      use_agent: bool = False,
                      initial_cash: float = 100000):
        """
        运行单个实验
        
        Args:
            experiment_name: 实验名称
            scorer: 评分函数
            use_agent: 是否使用 Agent
            initial_cash: 初始资金
        """
        print(f"\n{'='*60}")
        print(f"运行实验: {experiment_name}")
        print(f"使用 Agent: {use_agent}")
        print(f"{'='*60}")
        
        # 准备面板
        if 'alpha_132' in self.panel_with_alpha132.columns:
            panel = self.panel_with_alpha132.copy()
        else:
            panel = self.panel.copy()
        
        # 准备候选过滤器
        if use_agent:
            candidate_filter = lambda group: agent_gate(v8_candidate_filter(group))
        else:
            candidate_filter = v8_candidate_filter
        
        # 运行回测
        result = run_scored_backtest(
            panel,
            scorer,
            top_n=5,
            initial_cash=initial_cash,
            market_filter=True,
            retention_multiple=3,
            universe_size=1000,
            candidate_filter=candidate_filter
        )
        
        # 保存结果
        self.results[experiment_name] = {
            'metrics': result['metrics'],
            'trades': result['trades'],
            'use_agent': use_agent
        }
        
        # 打印结果
        metrics = result['metrics']
        print(f"\n实验结果:")
        print(f"总收益: {metrics['total_return']:.2%}")
        print(f"年化收益: {metrics['annual_return']:.2%}")
        print(f"夏普比率: {metrics['sharpe']:.4f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"胜率: {metrics['win_rate']:.2%}")
        print(f"交易次数: {metrics['closed_trades']}")
        
        return result
    
    def run_all_experiments(self, new_factor_names: List[str] = None,
                           initial_cash: float = 100000):
        """
        运行所有 6 个实验
        
        Args:
            new_factor_names: 新因子名称列表
            initial_cash: 初始资金
        """
        print("\n" + "="*60)
        print("开始运行所有实验")
        print("="*60)
        
        # 1. Baseline: Alpha191 (不带 Agent)
        self.run_experiment(
            "Baseline (Alpha191)",
            v8_scorer,
            use_agent=False,
            initial_cash=initial_cash
        )
        
        # 2. Baseline + Agent
        self.run_experiment(
            "Baseline + Agent",
            v8_scorer,
            use_agent=True,
            initial_cash=initial_cash
        )
        
        # 3. New Factors (不带 Agent)
        if new_factor_names:
            self.create_new_factor_scorer(new_factor_names)
            self.run_experiment(
                "New Factors",
                self.new_factor_scorer,
                use_agent=False,
                initial_cash=initial_cash
            )
            
            # 4. New Factors + Agent
            self.run_experiment(
                "New Factors + Agent",
                self.new_factor_scorer,
                use_agent=True,
                initial_cash=initial_cash
            )
            
            # 5. Hybrid (不带 Agent)
            self.create_hybrid_scorer(alpha132_weight=0.3, new_factor_weight=0.7)
            self.run_experiment(
                "Hybrid (Alpha191 + New)",
                self.hybrid_scorer,
                use_agent=False,
                initial_cash=initial_cash
            )
            
            # 6. Hybrid + Agent
            self.run_experiment(
                "Hybrid + Agent",
                self.hybrid_scorer,
                use_agent=True,
                initial_cash=initial_cash
            )
        
        print("\n" + "="*60)
        print("所有实验完成")
        print("="*60)
    
    def generate_comparison_report(self, output_file: str = None):
        """
        生成对比报告
        
        Args:
            output_file: 输出文件路径
        """
        print("\n生成对比报告...")
        
        # 汇总所有实验结果
        comparison_data = []
        
        for exp_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                '实验': exp_name,
                '使用 Agent': '是' if result['use_agent'] else '否',
                '总收益': f"{metrics['total_return']:.2%}",
                '年化收益': f"{metrics['annual_return']:.2%}",
                '夏普比率': f"{metrics['sharpe']:.4f}",
                '最大回撤': f"{metrics['max_drawdown']:.2%}",
                '胜率': f"{metrics['win_rate']:.2%}",
                '交易次数': metrics['closed_trades']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 计算 Agent 改善效果
        agent_improvement = []
        
        for exp_name in self.results.keys():
            if '+ Agent' in exp_name:
                base_name = exp_name.replace(' + Agent', '')
                if base_name in self.results:
                    base_metrics = self.results[base_name]['metrics']
                    agent_metrics = self.results[exp_name]['metrics']
                    
                    # 计算改善幅度
                    return_improvement = (
                        agent_metrics['total_return'] - 
                        base_metrics['total_return']
                    )
                    sharpe_improvement = (
                        agent_metrics['sharpe'] - 
                        base_metrics['sharpe']
                    )
                    drawdown_improvement = (
                        agent_metrics['max_drawdown'] - 
                        base_metrics['max_drawdown']
                    )
                    
                    agent_improvement.append({
                        '实验': exp_name,
                        '收益改善': f"{return_improvement:.2%}",
                        '夏普改善': f"{sharpe_improvement:.4f}",
                        '回撤改善': f"{drawdown_improvement:.2%}"
                    })
        
        improvement_df = pd.DataFrame(agent_improvement)
        
        # 生成报告文本
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("因子对比实验报告")
        report_lines.append("="*60)
        report_lines.append(f"实验时间: {self.start_date} 至 {self.end_date}")
        report_lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("一、实验结果对比")
        report_lines.append("-"*60)
        report_lines.append(comparison_df.to_string(index=False))
        report_lines.append("")
        
        if not improvement_df.empty:
            report_lines.append("二、Agent 改善效果")
            report_lines.append("-"*60)
            report_lines.append(improvement_df.to_string(index=False))
            report_lines.append("")
        
        # 分析结论
        report_lines.append("三、分析结论")
        report_lines.append("-"*60)
        
        # 找出最佳实验
        best_exp = max(
            self.results.keys(),
            key=lambda x: self.results[x]['metrics']['sharpe']
        )
        best_sharpe = self.results[best_exp]['metrics']['sharpe']
        
        report_lines.append(f"最佳实验: {best_exp}")
        report_lines.append(f"最佳夏普比率: {best_sharpe:.4f}")
        report_lines.append("")
        
        # 分析 Agent 效果
        agent_exps = [k for k in self.results.keys() if '+ Agent' in k]
        if agent_exps:
            avg_return_improvement = 0
            avg_sharpe_improvement = 0
            count = 0
            
            for exp_name in agent_exps:
                base_name = exp_name.replace(' + Agent', '')
                if base_name in self.results:
                    base_metrics = self.results[base_name]['metrics']
                    agent_metrics = self.results[exp_name]['metrics']
                    
                    avg_return_improvement += (
                        agent_metrics['total_return'] - 
                        base_metrics['total_return']
                    )
                    avg_sharpe_improvement += (
                        agent_metrics['sharpe'] - 
                        base_metrics['sharpe']
                    )
                    count += 1
            
            if count > 0:
                avg_return_improvement /= count
                avg_sharpe_improvement /= count
                
                report_lines.append(f"Agent 平均收益改善: {avg_return_improvement:.2%}")
                report_lines.append(f"Agent 平均夏普改善: {avg_sharpe_improvement:.4f}")
                report_lines.append("")
                
                if avg_sharpe_improvement > 0:
                    report_lines.append("结论: Agent 对策略有显著改善效果")
                else:
                    report_lines.append("结论: Agent 对策略改善效果有限")
        
        report_text = "\n".join(report_lines)
        
        print(report_text)
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n报告已保存到: {output_file}")
        
        return comparison_df, improvement_df


def run_comparison_experiment(data_dir: str,
                             start_date: str,
                             end_date: str,
                             top_n_factors: int = 5,
                             output_file: str = None):
    """
    运行完整的对比实验
    
    Args:
        data_dir: 数据目录
        start_date: 开始日期
        end_date: 结束日期
        top_n_factors: 选择的新因子数量
        output_file: 输出文件路径
    """
    # 创建实验
    experiment = ComparisonExperiment(data_dir, start_date, end_date)
    
    # 加载数据
    experiment.load_data()
    
    # 准备新因子
    experiment.prepare_new_factors()
    
    # 计算新因子
    experiment.compute_new_factors()
    
    # 分析新因子 IC
    ic_results = experiment.analyze_new_factors_ic(forward_days=5)
    
    # 选择最佳因子
    best_factors = experiment.select_best_factors(top_n=top_n_factors)
    
    # 运行所有实验
    experiment.run_all_experiments(new_factor_names=best_factors)
    
    # 生成报告
    comparison_df, improvement_df = experiment.generate_comparison_report(output_file)
    
    return experiment, comparison_df, improvement_df
