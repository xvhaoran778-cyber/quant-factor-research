import inspect
import os
import re
from functools import lru_cache

os.environ.setdefault("TA_CN_MODE", "LONG")

from ta_cn.alphas import alpha191


CURATED_EXPLANATIONS = {
    1: "观察成交量变化与日内涨跌之间是否背离。量价关系越反常，因子绝对值通常越明显。",
    2: "衡量收盘价在当日最高价和最低价之间的位置变化，可理解为多空力量失衡的变化。",
    3: "累计近期上涨或下跌时的有效价格移动，用来描述连续买卖压力。",
    4: "比较短期与中期均价、波动和成交量，判断价格是否偏离常态区间。",
    5: "观察成交量排序与最高价排序的相关性，捕捉短期量价同步或背离。",
    6: "观察开盘价与最高价组合在四日内的方向变化，偏向短期反转信号。",
    7: "结合成交量变化与成交均价相对收盘价的偏离，衡量短期交易拥挤。",
    8: "跟踪由最高价、最低价和成交均价合成的价格重心变化。",
    9: "把价格重心变化、振幅和成交量结合，近似刻画单位成交量推动价格的力量。",
    10: "下跌时使用收益波动、上涨时使用价格，寻找近期极端状态。",
    11: "将收盘价在日内区间的位置乘以成交量，累计衡量主动买卖压力。",
    12: "同时比较开盘价与近期成交均价、收盘价与当日成交均价的偏离。",
    13: "比较最高价与最低价的几何平均值和成交均价，观察成交重心偏离。",
    14: "当前收盘价减去五个交易日前收盘价，是直观的五日价格动量。",
    15: "今日开盘价相对昨日收盘价的跳空幅度。",
    18: "当前收盘价相对五日前的价格比率，反映短周期趋势。",
    20: "六日价格涨跌幅，数值越大表示近期上涨越强。",
}


def _formula(func) -> str:
    doc = inspect.getdoc(func) or ""
    return re.sub(r"^Alpha\d+\s*", "", doc.splitlines()[0]).strip()


def _category(formula: str) -> str:
    upper = formula.upper()
    if "VOLUME" in upper and "CORR" in upper:
        return "量价关系"
    if "STD" in upper or "VAR" in upper:
        return "波动率"
    if "HIGH" in upper and "LOW" in upper:
        return "价格位置"
    if "VOLUME" in upper or "VWAP" in upper or "AMOUNT" in upper:
        return "成交行为"
    if "DELTA" in upper or "DELAY" in upper or "RET" in upper:
        return "动量与反转"
    if "BANCHMARK" in upper or "MKT" in upper:
        return "市场相对强弱"
    return "价格趋势"


def _plain_explanation(number: int, formula: str, category: str) -> str:
    if number in CURATED_EXPLANATIONS:
        return CURATED_EXPLANATIONS[number]
    windows = sorted(set(re.findall(r"(?<![A-Z])\d{1,3}(?!\d)", formula)), key=int)
    window_text = f"，主要观察 {', '.join(windows[:3])} 个交易日附近的变化" if windows else ""
    descriptions = {
        "量价关系": "把价格变化和成交量变化放在一起比较，用来寻找量价同步、背离或交易拥挤",
        "波动率": "观察价格或收益的波动程度及其变化，识别风险扩张或收缩",
        "价格位置": "观察收盘价、最高价和最低价之间的相对位置，近似描述多空力量",
        "成交行为": "利用成交量或成交均价描述市场参与热度和资金推动强弱",
        "动量与反转": "比较当前价格与过去价格，判断短期趋势延续还是反转",
        "市场相对强弱": "比较个股和市场指数的表现，寻找相对强势或弱势股票",
        "价格趋势": "通过近期价格序列的统计特征描述短期趋势",
    }
    return descriptions[category] + window_text + "。公式较复杂时，先关注排序结果，不必手算。"


UNSUPPORTED_FACTORS = {
    30: "缺少真实 SMB/HML 历史数据，伪造为零会产生无意义结果",
}


@lru_cache
def factor_catalog() -> list[dict]:
    catalog = []
    for number in range(1, 192):
        func = getattr(alpha191, f"alpha_{number:03d}")
        formula = _formula(func)
        inputs = [name for name in inspect.signature(func).parameters if name != "kwargs"]
        category = _category(formula)
        unsupported_reason = UNSUPPORTED_FACTORS.get(number)
        catalog.append({
            "id": f"alpha_{number:03d}", "number": number, "name": f"Alpha {number:03d}",
            "category": category, "formula": formula,
            "explanation": _plain_explanation(number, formula, category), "inputs": inputs,
            "default_direction": "normal",
            "source": "国泰君安《基于短周期价量特征的多因子选股体系》(2017)",
            "supported": unsupported_reason is None,
            "unsupported_reason": unsupported_reason,
        })
    return catalog
