# GRAVITY_TEST 本文件是用户自定义策略

import pandas as pd
import numpy as np

def generate_signals(df):
    df = df.copy()
    # 计算EXPMA（指数移动平均），使用ewm(span)实现
    df['expma5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['expma12'] = df['close'].ewm(span=12, adjust=False).mean()
    # 初始化信号列
    df['signal'] = 0
    # 上穿买入：当前expma5 > expma12 且 前一周期expma5 <= expma12
    df.loc[(df['expma5'] > df['expma12']) & (df['expma5'].shift(1) <= df['expma12'].shift(1)), 'signal'] = 1
    # 下穿卖出：当前expma5 < expma12 且 前一周期expma5 >= expma12
    df.loc[(df['expma5'] < df['expma12']) & (df['expma5'].shift(1) >= df['expma12'].shift(1)), 'signal'] = -1
    return df

# STRATEGY_NAME = "expma"
