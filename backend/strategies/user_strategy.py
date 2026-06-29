"""用户自定义策略加载器"""

import os
import re
import sys
import importlib.util
from typing import Dict, List, Optional
from loguru import logger


class UserStrategy:
    """用户自定义策略"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.name = self._extract_name(filepath)
        self.description = self._extract_desc(filepath)
        self.module = None
        self._loaded = False
    
    def _extract_name(self, filepath: str) -> str:
        """从文件提取策略名称"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            match = re.search(r'STRATEGY_NAME\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
            match = re.search(r'STRATEGY_NAME\s*=\s*\'([^\']+)\'', content)
            if match:
                return match.group(1)
        except:
            pass
        return os.path.splitext(os.path.basename(filepath))[0].replace('_', ' ').title()
    
    def _extract_desc(self, filepath: str) -> str:
        """从文件提取策略描述"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            match = re.search(r'STRATEGY_DESC\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
            match = re.search(r'STRATEGY_DESC\s*=\s*\'([^\']+)\'', content)
            if match:
                return match.group(1)
        except:
            pass
        return ""
    
    def load(self) -> bool:
        """加载策略模块"""
        if self._loaded and self.module:
            return True
        
        try:
            # 读取文件内容
            with open(self.filepath, 'r') as f:
                code = f.read()
            
            # 找到GRAVITY_TEST之后的内容
            marker = '# GRAVITY_TEST'
            if marker in code:
                # 切掉整行 GRAVITY_TEST 注释（包括后面的中文）
                lines = code.split('\n')
                user_lines = []
                found = False
                for line in lines:
                    if marker in line:
                        found = True
                        # 取该行 marker 之后可能的中文注释，跳过
                        continue
                    if found:
                        user_lines.append(line)
                user_code = '\n'.join(user_lines)
            else:
                user_code = code
            
            # 动态编译执行 - 注入所需依赖
            namespace = {
                '__builtins__': __builtins__,
                'pd': __import__('pandas'),
                'np': __import__('numpy'),
                'pandas': __import__('pandas'),
                'numpy': __import__('numpy'),
                'go': __import__('plotly.graph_objects'),
                'px': __import__('plotly.express')
            }
            exec(compile(user_code, self.filepath, 'exec'), namespace)
            
            if 'generate_signals' in namespace:
                self.module = namespace
                self._loaded = True
                return True
            else:
                logger.warning(f"策略 {self.name} 缺少 generate_signals 函数")
                return False
                
        except Exception as e:
            logger.error(f"加载策略 {self.name} 失败: {e}")
            return False
    
    def get_description(self) -> str:
        """获取策略描述"""
        return f"""
【用户策略: {self.name}】

{self.description}

文件: {self.filename}
路径: {self.filepath}
"""
    
    def backtest(self, df, initial_capital: float = 1000000) -> Dict:
        """回测策略"""
        import pandas as pd
        import numpy as np
        
        if not self.load():
            return {'total_return': 0, 'total_trades': 0, 'trades': [], 'equity_curve': [], 
                    'strategy_name': self.name, 'error': '策略加载失败'}
        
        df = df.copy()
        
        try:
            df = self.module['generate_signals'](df)
        except Exception as e:
            logger.error(f"策略 {self.name} generate_signals 执行失败: {e}")
            return {'total_return': 0, 'total_trades': 0, 'trades': [], 'equity_curve': [],
                    'strategy_name': self.name, 'error': f'代码执行错误: {e}'}
        
        if 'signal' not in df.columns:
            return {'total_return': 0, 'total_trades': 0, 'trades': [], 'equity_curve': [],
                    'strategy_name': self.name, 'error': 'generate_signals 没有返回 signal 列'}
        
        # 统计信号分布
        signal_counts = {'buy': int((df['signal'] == 1).sum()), 
                        'sell': int((df['signal'] == -1).sum()),
                        'hold': int((df['signal'] == 0).sum())}
        
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            if row['signal'] == 1 and position == 0:
                buy_price = row['close']
                shares = int(capital * 0.95 / buy_price / 100) * 100
                if shares > 0:
                    cost = shares * buy_price * 1.0003
                    capital -= cost
                    position = shares
                    trades.append({'type': 'buy', 'price': buy_price, 'shares': shares, 'idx': i,
                                  'date': str(row.get('date', i))})
            
            elif row['signal'] == -1 and position > 0:
                sell_price = row['close']
                revenue = position * sell_price * 0.9997
                capital += revenue
                trades.append({'type': 'sell', 'price': sell_price, 'shares': position, 'idx': i,
                              'date': str(row.get('date', i))})
                position = 0
            
            equity = capital + position * row['close']
            equity_curve.append(equity)
        
        # 最终平仓
        if position > 0 and len(df) > 0:
            final_price = df.iloc[-1]['close']
            capital += position * final_price * 0.9997
            trades.append({'type': 'sell', 'price': final_price, 'shares': position, 'idx': len(df)-1,
                          'date': str(df.iloc[-1].get('date', len(df)-1))})
            position = 0
            equity_curve[-1] = capital
        
        equity_series = pd.Series(equity_curve) if equity_curve else pd.Series([initial_capital])
        total_return = (equity_series.iloc[-1] / initial_capital - 1) * 100 if len(equity_series) > 0 else 0
        
        return {
            'strategy_name': self.name,
            'initial_capital': initial_capital,
            'final_equity': equity_series.iloc[-1] if len(equity_series) > 0 else initial_capital,
            'total_return': total_return,
            'total_trades': len(trades),
            'trades': trades,
            'equity_curve': equity_curve,
            'signal_counts': signal_counts
        }


class UserStrategyLoader:
    """用户策略加载器"""
    
    def __init__(self, strategies_dir: str = None):
        if strategies_dir is None:
            strategies_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'user_strategies')
        self.strategies_dir = strategies_dir
        os.makedirs(self.strategies_dir, exist_ok=True)
    
    def list_strategies(self) -> List[UserStrategy]:
        """列出所有用户策略"""
        strategies = []
        
        if not os.path.exists(self.strategies_dir):
            return strategies
        
        for filename in os.listdir(self.strategies_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                filepath = os.path.join(self.strategies_dir, filename)
                strategy = UserStrategy(filepath)
                strategies.append(strategy)
        
        return strategies
    
    def get_strategy(self, filename: str) -> Optional[UserStrategy]:
        """获取指定策略"""
        filepath = os.path.join(self.strategies_dir, filename)
        if os.path.exists(filepath) and not filename.startswith('_'):
            return UserStrategy(filepath)
        return None
    
    def save_strategy(self, filename: str, code: str, name: str = None, desc: str = None) -> str:
        """保存策略文件"""
        if not filename.endswith('.py'):
            filename += '.py'
        
        filepath = os.path.join(self.strategies_dir, filename)
        
        # 构建完整文件内容
        header = '# GRAVITY_TEST 本文件是用户自定义策略\n\n'
        
        full_code = header + code
        
        if name:
            full_code += f'\n\n# STRATEGY_NAME = "{name}"\n'
        if desc:
            full_code += f'# STRATEGY_DESC = "{desc}"\n'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_code)
        
        logger.info(f"策略已保存: {filepath}")
        return filepath
    
    def delete_strategy(self, filename: str) -> bool:
        """删除策略文件"""
        filepath = os.path.join(self.strategies_dir, filename)
        if os.path.exists(filepath) and not filename.startswith('_'):
            os.remove(filepath)
            logger.info(f"策略已删除: {filepath}")
            return True
        return False
