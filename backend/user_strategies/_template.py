# GRAVITY_TEST 本文件将被用户自定义策略替换
# 请勿删除此标记，系统依赖它来定位用户代码

def generate_signals(df):
    """生成交易信号 - 在这里编写你的策略逻辑
    
    Args:
        df: pandas.DataFrame, 包含: open, high, low, close, volume
        
    Returns:
        df: 添加了'signal'列的DataFrame
            1 = 买入, -1 = 卖出, 0 = 持有
    """
    df = df.copy()
    
    # 在此编写策略逻辑
    # 示例: 当收盘价上穿5日均线时买入
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    
    df['signal'] = 0
    df.loc[(df['ma5'] > df['ma10']) & (df['ma5'].shift(1) <= df['ma10'].shift(1)), 'signal'] = 1
    df.loc[(df['ma5'] < df['ma10']) & (df['ma5'].shift(1) >= df['ma10'].shift(1)), 'signal'] = -1
    
    return df

# STRATEGY_NAME = "我的均线策略"
# STRATEGY_DESC = "短期均线上穿长期均线时买入"
