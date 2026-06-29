"""A股交易规则"""
from datetime import datetime, time, timedelta

# 交易时段
MORNING_START = time(9, 30)
MORNING_END = time(11, 30)
AFTERNOON_START = time(13, 0)
AFTERNOON_END = time(15, 0)

# 手续费率
COMMISSION_RATE = 0.00025  # 万分之2.5
STAMP_TAX_RATE = 0.0005    # 万分之5（仅卖出）
MIN_COMMISSION = 5.0       # 最低5元

# 涨跌停
MAIN_BOARD_LIMIT = 0.10    # 主板±10%
CHINEXT_LIMIT = 0.20       # 创业板/科创板±20%

# 最小交易单位(股)
LOT_SIZE = 100

# T+1(买入当天不能卖出)
T_PLUS_ONE = True

def is_trading_time() -> bool:
    """是否在交易时间"""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return (MORNING_START <= t <= MORNING_END) or (AFTERNOON_START <= t <= AFTERNOON_END)

def is_trading_day() -> bool:
    """是否交易日(简化：周一至周五)"""
    return datetime.now().weekday() < 5

def can_buy_today(buy_date_str: str) -> bool:
    """判断T+1规则：今天买入的明天才能卖出"""
    try:
        buy_date = datetime.strptime(buy_date_str, "%Y-%m-%d")
        return datetime.now() >= buy_date + timedelta(days=1)
    except:
        return True

def calc_commission(amount: float, is_sell: bool = False) -> float:
    """计算手续费"""
    comm = max(amount * COMMISSION_RATE, MIN_COMMISSION)
    if is_sell:
        comm += amount * STAMP_TAX_RATE
    return comm

def calc_shares(amount: float, price: float) -> int:
    """计算可买股数(100股整数倍)"""
    return int(amount / price / LOT_SIZE) * LOT_SIZE

def filter_by_price_limit(code: str, price: float, prev_close: float) -> bool:
    """检查是否涨跌停"""
    if code.startswith(('300', '301', '688')):
        limit = CHINEXT_LIMIT
    else:
        limit = MAIN_BOARD_LIMIT
    change = abs(price / prev_close - 1) if prev_close > 0 else 0
    return change < limit * 0.95  # 留5%余量避免买在涨停板
