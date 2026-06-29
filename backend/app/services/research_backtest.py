from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from app.services.backtest import TradingRules, trade_cost
from app.services.market_store import ParquetMarketStore
from app.services.market_store import is_excluded_security_name


PRESETS = {
    "alpha191_132_173_research": {
        "name": "Alpha191 双因子研究版",
        "summary": "从国泰君安 Alpha191 全库训练期筛选并去相关，使用 Alpha132 与 Alpha173 的反向排名等权选股。",
        "formula": "50% Alpha132反向（20日平均成交额相对较低）+ 50% Alpha173反向（平滑价格趋势相对较低）",
        "basis": "191个因子中190个成功计算；组合成员、方向和权重只由2023-2024训练期决定，2025-2026冻结验收累计9.30%。",
        "risk": "balanced", "risk_label": "研究观察",
        "market_fit": "大盘未处于明确下行，且高流动性股票内部存在成交热度与平滑趋势均值回归",
        "holding": "周五完整数据评分，下一交易日开盘买入前5名；保留排名前10的原持仓",
        "risk_note": "训练期累计105.15%、夏普1.32，但验收期仅9.30%、夏普0.45，衰减明显，不应直接实盘。",
        "default_top_n": 5, "tags": ["Alpha191", "多因子", "训练隔离", "验证衰减", "研究版"],
        "market_filter": True, "retention_multiple": 2, "universe_size": 800,
        "dedicated_engine": "alpha191_research", "validation_status": "research_only",
        "validation_summary": {"training_return": 1.051529, "training_sharpe": 1.3163, "holdout_return": 0.093011, "holdout_sharpe": 0.4468, "holdout_max_drawdown": -0.168741},
    },
    "alpha191_173_174_083_v2": {
        "name": "Alpha191 三因子稳健 V2",
        "summary": "旧短样本研究冠军；在统一后复权价格、固定800股股票池和2012-2026长周期复核后失效。",
        "formula": "风险调整权重：Alpha173反向 3.444 + Alpha174反向 2.837 + Alpha083正向 1.207",
        "basis": "旧2025-2026短验收曾显示42.77%；同口径锁箱期2024-2026为-15.41%、夏普-0.29，不可继续作为有效策略。",
        "risk": "aggressive", "risk_label": "长周期复核失效",
        "market_fit": "大盘未明确下行，高流动性股票内部的平滑趋势、上涨波动与量价协方差共同有效",
        "holding": "周五评分、下个交易日开盘执行；持有前3名，原持仓在前9名内可保留",
        "risk_note": "保留用于对照，不建议模拟或实盘；短样本高收益未通过长周期复核。",
        "default_top_n": 3, "tags": ["Alpha191", "三因子", "5000组搜索", "逐年复核", "冻结验收"],
        "market_filter": True, "retention_multiple": 3, "universe_size": 800,
        "dedicated_engine": "alpha191_research", "validation_status": "long_cycle_failed",
        "validation_summary": {"development_return": 0.098411, "development_sharpe": 0.1589, "validation_return": -0.009356, "validation_sharpe": 0.0662, "holdout_return": -0.154126, "holdout_sharpe": -0.2907, "holdout_max_drawdown": -0.240335},
    },
    "alpha191_042_061_095_v3": {
        "name": "Alpha191 长周期三因子 V3",
        "summary": "在2012至2023多轮行情中筛选，使用两个量价稳定性因子和一个低成交额波动因子进行周频选股。",
        "formula": "等权：Alpha042正向 + Alpha061正向 + Alpha095反向",
        "basis": "从100,000组配置中筛选，600组快速回测、180组完整撮合、60组六窗口复核；统一后复权口径下2024-2026锁箱累计40.65%。",
        "risk": "balanced", "risk_label": "长周期研究候选",
        "market_fit": "大盘未明确下行，低成交额波动且量价关系较稳定的高流动性股票占优",
        "holding": "周五评分、下一交易日开盘执行；等权持有前8名，每周重新排名",
        "risk_note": "2016-2019偏弱且2026年上半年亏损；锁箱夏普0.66、最大回撤15.82%，只适合模拟研究。",
        "default_top_n": 8, "tags": ["Alpha191", "长周期", "完整牛熊", "十万组搜索", "锁箱验证"],
        "market_filter": True, "retention_multiple": 1, "universe_size": 800,
        "dedicated_engine": "alpha191_research", "validation_status": "long_cycle_research",
        "validation_summary": {"development_return": 2.612343, "development_sharpe": 0.8215, "validation_return": 0.715619, "validation_sharpe": 0.9267, "holdout_return": 0.406472, "holdout_annual_return": 0.152451, "holdout_sharpe": 0.6555, "holdout_max_drawdown": -0.158174},
    },
    "reversal_5d": {
        "name": "短期反转增强", "summary": "偏好近 5 日超跌、成交活跃且波动不过高的股票。",
        "formula": "-5日收益 - 0.25 × 20日波动率", "basis": "中国 A 股研究中短期反转较稳健；使用流动性过滤降低不可交易样本影响。",
        "risk": "balanced", "risk_label": "均衡", "market_fit": "震荡或快速回调后的修复行情", "holding": "每周调仓，持有前 5 名",
        "risk_note": "连续下跌阶段可能反复抄底，必须观察最大回撤。", "default_top_n": 5, "tags": ["反转", "流动性", "低波动约束"],
    },
    "low_volatility": {
        "name": "低波动稳健", "summary": "在活跃股票中选择近 20 日波动率最低的一组。",
        "formula": "-20日收益波动率", "basis": "风险类异常与低波动效应在中国 A 股文献中有较多证据。",
        "risk": "steady", "risk_label": "稳健", "market_fit": "震荡、弱市和风险偏好下降阶段", "holding": "每周调仓，持有前 5 名",
        "risk_note": "牛市快速上涨时可能明显落后高弹性股票。", "default_top_n": 5, "tags": ["低波动", "防守", "高流动性"],
    },
    "volume_trend": {
        "name": "量价趋势", "summary": "结合 20 日动量、60 日均线趋势和近期量能，并惩罚高波动。",
        "formula": "20日收益 + 60日趋势 + 0.2 × 量比 - 0.5 × 20日波动率", "basis": "作为趋势跟随对照组；中国市场经典中期动量证据较弱，因此加入量能与风险约束。",
        "risk": "balanced", "risk_label": "均衡", "market_fit": "指数和行业趋势较清晰的行情", "holding": "每周调仓，持有前 5 名",
        "risk_note": "横盘行情容易频繁换股并被手续费侵蚀。", "default_top_n": 5, "tags": ["动量", "均线", "量价"],
    },
    "low_vol_trend": {
        "name": "低波动趋势增强", "summary": "选择中期趋势向上且波动较低的高流动性股票，降低追逐高波动热点的影响。",
        "formula": "0.45×60日趋势 + 0.25×20日动量 - 0.30×20日波动率", "basis": "根据聚宽公开社区常见的低波动与趋势过滤思路改写，未复制社区策略代码。",
        "risk": "steady", "risk_label": "稳健", "market_fit": "缓慢上行或结构性行情", "holding": "每周调仓，持有前 5 名",
        "risk_note": "趋势突然反转时信号会有滞后。", "default_top_n": 5, "tags": ["低波动", "趋势", "回撤控制"],
    },
    "filtered_reversal": {
        "name": "短期反转过滤", "summary": "寻找 5 日超跌，但保留中期趋势和低波动过滤，减少接住持续下跌股票。",
        "formula": "-0.60×5日收益 + 0.15×60日趋势 - 0.25×20日波动率", "basis": "根据聚宽公开的均值回归与风险过滤思路改写，适配当前日线数据。",
        "risk": "balanced", "risk_label": "均衡", "market_fit": "中期趋势尚未破坏的短期回调", "holding": "每周调仓，持有前 5 名",
        "risk_note": "趋势过滤不能完全避开基本面恶化的股票。", "default_top_n": 5, "tags": ["超跌", "趋势过滤", "均值回归"],
    },
    "price_volume_momentum": {
        "name": "量价趋势轮动", "summary": "用中期动量和趋势确认方向，再用温和放量确认资金参与。",
        "formula": "0.45×20日动量 + 0.35×60日趋势 + 0.20×量比 - 0.20×波动率", "basis": "根据聚宽公开的量价趋势与轮动思路改写，采用下一交易日开盘撮合。",
        "risk": "balanced", "risk_label": "均衡", "market_fit": "板块轮动和放量上行行情", "holding": "每周调仓，持有前 5 名",
        "risk_note": "放量也可能来自冲高出货，需要结合回撤观察。", "default_top_n": 5, "tags": ["量价", "轮动", "趋势确认"],
    },
    "oversold_rebound_aggressive": {
        "name": "极限超跌反弹", "summary": "寻找 5 日跌幅靠前、跌破布林中轨且 RSI 偏低的活跃股票，博取快速修复。",
        "formula": "0.45×超跌 + 0.25×布林偏离 + 0.20×低RSI + 0.10×低波动", "basis": "参考 A 股短期反转研究构建；这是高风险实验模型，不等同于稳定套利。",
        "risk": "aggressive", "risk_label": "激进", "market_fit": "恐慌回落后的快速修复行情", "holding": "每周调仓，集中持有前 3 名",
        "risk_note": "最容易遇到下跌中继，可能连续数周亏损。", "default_top_n": 3, "tags": ["超跌", "RSI", "布林带", "高风险"],
    },
    "breakout_acceleration": {
        "name": "放量突破加速", "summary": "追踪 20 日强势、站上 60 日趋势且近期明显放量的高弹性股票。",
        "formula": "0.35×20日动量 + 0.30×60日趋势 + 0.25×量比 + 0.10×波动弹性", "basis": "结合趋势延续与成交量确认，用作强势市场中的进攻型对照组。",
        "risk": "aggressive", "risk_label": "激进", "market_fit": "指数强势、热点持续和风险偏好上升阶段", "holding": "每周调仓，集中持有前 3 名",
        "risk_note": "追高风险明显，热点退潮时回撤可能很快。", "default_top_n": 3, "tags": ["突破", "放量", "高弹性", "高风险"],
    },
    "close_strength_continuation": {
        "name": "尾盘强势延续（日线版）", "summary": "用当日和近 5 日强度、量能与中期趋势模拟尾盘强势股的次日延续。",
        "formula": "0.35×当日强度 + 0.25×5日动量 + 0.25×量比 + 0.15×60日趋势", "basis": "由公开尾盘动量思路改写为日线可验证版本，不使用分钟级或未来数据。",
        "risk": "aggressive", "risk_label": "激进", "market_fit": "短线情绪活跃且强势股有延续性的行情", "holding": "每周调仓，集中持有前 3 名",
        "risk_note": "日线无法复刻真实尾盘盘口，隔夜跳空和涨跌停风险较高。", "default_top_n": 3, "tags": ["尾盘强度", "短线", "量能", "高风险"],
    },
    "weekly_defensive_reversal": {
        "name": "周频低波动反转", "summary": "在大盘多头阶段，买入短期回调、波动较低且中期趋势未破坏的高流动性股票。",
        "formula": "40%短期反转 + 30%低波动 + 20%中期趋势 + 10%流动性", "basis": "A 股异常研究对风险和短期反转的证据相对较强；加入大盘趋势过滤减少系统性下跌暴露。",
        "risk": "steady", "risk_label": "稳健", "market_fit": "大盘中期向上、个股短期回调", "holding": "周五评分，下个交易日换入前 5 名",
        "risk_note": "大盘过滤会错过 V 型反弹，震荡市也可能频繁空仓。", "default_top_n": 5, "tags": ["周频", "低波动", "反转", "大盘过滤"], "market_filter": True, "retention_multiple": 2,
    },
    "weekly_trend_quality_proxy": {
        "name": "周频稳健趋势", "summary": "在大盘趋势向上时，选择中期趋势强、20 日动量较好且波动较低的活跃股票。",
        "formula": "35%中期趋势 + 30%动量 + 25%低波动 + 10%流动性", "basis": "以低波动和流动性作为质量代理，避免仅追逐高波动热门股。",
        "risk": "balanced", "risk_label": "均衡", "market_fit": "指数和行业趋势持续上行", "holding": "周五评分，下个交易日换入前 5 名",
        "risk_note": "趋势信号有滞后，快速风格切换时可能连续换股。", "default_top_n": 5, "tags": ["周频", "趋势", "低波动", "大盘过滤"], "market_filter": True, "retention_multiple": 2,
    },
    "weekly_multi_factor_guarded": {
        "name": "周频防守多因子", "summary": "把反转、趋势、低波动、温和量能和流动性合成一个评分，大盘转弱时全部空仓。",
        "formula": "25%反转 + 25%趋势 + 25%低波动 + 15%温和量能 + 10%流动性", "basis": "组合互补的交易与风险因子，降低单一因子周期性失效的影响。",
        "risk": "steady", "risk_label": "稳健", "market_fit": "大盘多头中的震荡与结构性轮动", "holding": "周五评分，下个交易日换入前 5 名",
        "risk_note": "严格空仓规则可能降低长期持仓时间，也不能消除个股风险。", "default_top_n": 5, "tags": ["周频", "多因子", "全进全出", "大盘过滤"], "market_filter": True, "retention_multiple": 2,
    },
    "chanlun_macd_td_proxy": {
        "name": "缠论 MACD 背驰代理", "summary": "把缠论二买/中枢回踩、MACD 背驰和 TD 衰竭思想改写成 A 股日线周频选股模型。",
        "formula": "30%中枢回踩 + 25%背驰修复 + 20%TD衰竭 + 15%趋势保护 + 10%温和放量",
        "basis": "参考 haigechanlun/chanlun_auto_trading 的开源缠论分型、笔、中枢和 MACD+TD 示例思路，按本系统日线数据重写为可回测评分；未接入其不开源实盘模型。",
        "risk": "aggressive", "risk_label": "激进", "market_fit": "大盘未明显下行，个股经过回调后出现止跌修复", "holding": "周五评分，下个交易日买入前 3 名",
        "risk_note": "这是日线近似版，不等同于原项目多周期实盘策略；背驰信号可能过早，弱市中会反复假反弹。",
        "default_top_n": 3, "tags": ["缠论", "MACD背驰", "TD衰竭", "中枢回踩", "大盘过滤"],
        "market_filter": True, "retention_multiple": 1,
        "source_url": "https://github.com/haigechanlun/chanlun_auto_trading",
    },
    "social_small_cap_quality_momentum": {
        "name": "社媒小市值质量动量代理", "summary": "把 X/Reddit/中文社区常见的小市值高收益思路改写为成交额反向、低波动和中期动量组合。",
        "formula": "35%低成交额代理小盘 + 25%20日动量 + 20%低波动 + 10%60日趋势 + 10%流动性底线",
        "basis": "参考 JoinQuant 小市值策略讨论、Qbot 小市值文档和 BigQuant 小市值+动量思路；当前系统没有真实市值/财务字段，因此使用成交额反向作为小盘代理。",
        "risk": "aggressive", "risk_label": "激进", "market_fit": "小盘活跃、大盘不处于明显下行阶段", "holding": "周五评分，下个交易日买入前 4 名",
        "risk_note": "小盘代理可能买到流动性较差的股票，实盘冲击成本会显著高于回测。",
        "default_top_n": 4, "tags": ["小市值代理", "动量", "低波动", "大盘过滤"], "market_filter": True, "retention_multiple": 2, "universe_size": 2500,
        "source_url": "https://github.com/UFund-Me/Qbot/blob/main/docs/02-%E7%BB%8F%E5%85%B8%E7%AD%96%E7%95%A5/01-%E8%82%A1%E7%A5%A8/%E5%B0%8F%E5%B8%82%E5%80%BC.md",
    },
    "small_cap_reversal_guarded": {
        "name": "小盘超跌反转代理", "summary": "在非熊市中寻找成交额偏小、短期超跌但中期趋势没有完全破坏的股票。",
        "formula": "35%小盘代理 + 30%5日超跌 + 20%趋势保护 + 15%低波动",
        "basis": "来自小市值长期超额和 A 股短期反转的公开讨论，加入大盘过滤和低波动约束以避免纯接飞刀。",
        "risk": "aggressive", "risk_label": "激进", "market_fit": "小盘风格占优、短期急跌后修复", "holding": "周五评分，下个交易日买入前 4 名",
        "risk_note": "连续下跌时会反复买入弱势股，胜率可能高但单次亏损较大。",
        "default_top_n": 4, "tags": ["小盘代理", "超跌", "均值回归", "大盘过滤"], "market_filter": True, "retention_multiple": 1, "universe_size": 2500,
    },
    "low_price_breakout_proxy": {
        "name": "低价窄幅突破代理", "summary": "参考低价股动量策略：过滤低价、窄幅震荡、放量且趋势转强的股票。",
        "formula": "30%20日动量 + 25%60日突破位置 + 20%温和放量 + 15%低波动 + 10%流动性",
        "basis": "参考 FinLab 低价股突破研究，将台股规则改写为 A 股日线代理：低价、窄幅区间、放量、趋势确认。",
        "risk": "aggressive", "risk_label": "激进", "market_fit": "题材活跃、低价股补涨行情", "holding": "周五评分，下个交易日买入前 5 名",
        "risk_note": "低价股容易受题材和流动性影响，必须排除 ST/退市并控制仓位。",
        "default_top_n": 5, "tags": ["低价股", "突破", "放量", "高风险"], "market_filter": True, "retention_multiple": 1, "universe_size": 1800,
        "source_url": "https://finlab.finance/blog/low-price-stock-quant-strategy",
    },
    "limit_up_continuation_proxy": {
        "name": "首板延续代理", "summary": "用日涨幅、量能和趋势近似社媒常见的 1 进 2/弱转强短线思路，但不买 ST。",
        "formula": "35%当日强度 + 25%5日强度 + 20%放量 + 10%趋势 + 10%低波动",
        "basis": "参考 JoinQuant 1进2策略解读与 CSDN 弱转强文章；本系统不用分钟盘口，只用已完成日线近似，且坚持排除 ST。",
        "risk": "aggressive", "risk_label": "激进", "market_fit": "连板情绪活跃、短线资金风险偏好高", "holding": "周五评分，下个交易日买入前 3 名",
        "risk_note": "日线近似无法确认真实封板质量，隔日高开回落和涨跌停买不到/卖不出风险很高。",
        "default_top_n": 3, "tags": ["首板", "弱转强", "放量", "短线"], "market_filter": True, "retention_multiple": 1, "universe_size": 2200,
        "source_url": "https://www.joinquant.com/post/386a26aba98a6df6c4258976362b006b",
    },
    "leader_relay_daily": {
        "name": "龙头接力实验（验证未通过）",
        "summary": "研究市场情绪和板块共振下的首板接力；90组训练参数与锁定留出集均亏损，仅供观察失败原因。",
        "formula": "30%板块涨停扩散 + 20%流动性 + 15%短期强度 + 15%健康量能 + 10%趋势 - 10%高波动",
        "basis": "依据涨跌停延迟价格发现与过度反应研究构建；只使用完整日线，次日开盘交易，并模拟涨停买不到、跌停卖不出。",
        "risk": "aggressive", "risk_label": "高风险实验", "market_fit": "指数中期向上、涨停家数较多且板块出现至少2只涨停",
        "holding": "日频扫描，最多2只，默认持有2个交易日；6%止损、16%止盈均在信号次日执行",
        "risk_note": "3万元回测：训练期-39.04%、夏普-0.66；锁定留出期-68.84%、夏普-2.16。不可用于模拟盘或实盘。",
        "default_top_n": 2, "tags": ["验证未通过", "龙头", "首板", "板块共振", "日频", "可成交约束"],
        "market_filter": True, "retention_multiple": 1, "universe_size": 2500,
        "source_url": "https://xbbjb.cufe.edu.cn/EN/Y2025/V0/I1/59",
        "dedicated_engine": "leader_relay_daily",
        "validation_status": "rejected",
        "validation_summary": {"training_return": -0.390355, "training_sharpe": -0.6641, "holdout_return": -0.688436, "holdout_sharpe": -2.1621},
    },
    "core_asset_momentum_rotation": {
        "name": "核心资产动量轮动代理", "summary": "把 ETF/核心资产轮动思路迁移到高流动性大盘股：只买趋势和动量最强的一小组。",
        "formula": "45%20日动量 + 25%60日趋势 + 20%风险调整动量 + 10%流动性",
        "basis": "参考 ETF 动量轮动研究和核心资产轮动文章；当前股票池没有 ETF 专属分类，因此用高流动性股票做核心资产代理。",
        "risk": "balanced", "risk_label": "均衡", "market_fit": "核心资产趋势清晰、指数未破位", "holding": "周五评分，下个交易日买入前 5 名",
        "risk_note": "股票代理不等同于 ETF 轮动，行业暴露更集中。",
        "default_top_n": 5, "tags": ["核心资产", "动量轮动", "高流动性", "大盘过滤"], "market_filter": True, "retention_multiple": 2, "universe_size": 350,
        "source_url": "https://pdf.dfcfw.com/pdf/H301_AP202412241641416114_1.pdf",
    },
    "dual_momentum_cash_guard": {
        "name": "二八动量空仓代理", "summary": "借鉴二八轮动：大盘转弱空仓，大盘转强时在高流动性股票中追踪动量最强者。",
        "formula": "40%20日动量 + 30%60日趋势 + 20%低波动 + 10%流动性；大盘下行空仓",
        "basis": "参考二八轮动和 ETF 轮动公开思路，强调先判断市场风险，再决定是否持仓。",
        "risk": "steady", "risk_label": "稳健", "market_fit": "指数趋势上行或结构性行情", "holding": "周五评分，下个交易日买入前 4 名",
        "risk_note": "空仓规则会错过 V 型反弹；强趋势后段也可能追高。",
        "default_top_n": 4, "tags": ["二八轮动", "空仓", "动量", "大盘过滤"], "market_filter": True, "retention_multiple": 2, "universe_size": 500,
        "source_url": "https://bigquant.com/wiki/doc/4Zo0mRaP6V",
    },

    "liquidity_strength_rotation": {
        "name": "高流动性强弱轮动", "summary": "基于近期验证结果：优先在成交额最高的一批股票中轮动强趋势、低波动标的。",
        "formula": "35%流动性 + 30%20日动量 + 20%低波动 + 15%60日趋势",
        "basis": "本系统 2023-2026 单因子验证中，高流动性因子收益和夏普靠前；该策略把流动性作为主因子，同时加入趋势和低波动避免只买大成交热门股。",
        "risk": "balanced", "risk_label": "均衡", "market_fit": "核心资产或大成交股票持续走强，大盘未明显破位", "holding": "周五评分，下个交易日买入前 5 名",
        "risk_note": "流动性强不等于一定上涨，拥挤交易退潮时可能回撤；大盘过滤仍可能滞后。",
        "default_top_n": 5, "tags": ["流动性", "强弱轮动", "低波动", "大盘过滤"], "market_filter": True, "retention_multiple": 2, "universe_size": 600,
    },
    "liquidity_strength_rotation_guarded": {
        "name": "高流动性强弱轮动优化版",
        "summary": "先限定在高流动性股票池，再选择趋势向上、波动较低、近期涨幅确认的股票，避免只追大成交热门股。",
        "formula": "45%60日趋势 + 25%20日动量 + 20%低波动 + 10%流动性；候选需 20日收益>2%、趋势>0、波动低于当周80%分位",
        "basis": "对原高流动性策略的回撤归因显示，主要亏损来自高成交热门股退潮和波动放大。参数搜索后采用高流动性股票池 + 低波动趋势评分，原策略不改动。",
        "risk": "balanced", "risk_label": "均衡优化", "market_fit": "高流动性核心股票仍有趋势，但市场波动开始升高的阶段",
        "holding": "周五评分，下个交易日买入前 4 名；持仓留存到前 12 名以内",
        "risk_note": "过滤更严会减少交易次数，可能错过 V 型反弹和强热点末段。",
        "default_top_n": 4, "tags": ["流动性", "优化版", "低波动趋势", "回撤控制"], "market_filter": True, "retention_multiple": 3, "universe_size": 1200,
    },
    "liquidity_strength_rotation_v1": {
        "name": "高流动性强弱轮动 V1",
        "summary": "保留原策略不变，以低波动为核心，使用8只核心卫星持仓、行业上限和高波动持仓盘中保护单。",
        "formula": "45%非线性低波动 + 20%趋势 + 15%动量 + 15%下行波动 + 5%流动性；单行业最多2只",
        "basis": "参数消融显示机械止损和高动量会损害收益；V1 仅对相对高波动持仓预挂5%保护单，并保留中证500与全A等权状态用于调仓解释。",
        "risk": "steady", "risk_label": "V1 防守增强", "market_fit": "上证未破位，且中证500或全A等权代理至少一个保持可参与状态",
        "holding": "每周持有8只，核心5只占95%；留存前16名；相对高波动持仓盘中跌5%保护单减半",
        "risk_note": "全样本参数较敏感，行业数据是当前快照而非历史时点分类；保护单仍可能在快速反弹前卖出。",
        "default_top_n": 8, "tags": ["V1", "日频止损", "行业分散", "下行波动", "多市场风控"],
        "market_filter": True, "retention_multiple": 1, "universe_size": 1200,
    },
    "low_vol_style_rotation": {
        "name": "低波动风格轮动", "summary": "在大盘可参与阶段，轮动到低波动、趋势不差、成交活跃的防守型股票。",
        "formula": "45%低波动 + 25%60日趋势 + 20%流动性 + 10%温和放量",
        "basis": "低波动因子在近期验证中回撤较小、夏普较稳，适合作为小资金先活下来再进攻的轮动底仓。",
        "risk": "steady", "risk_label": "稳健", "market_fit": "震荡市、弱复苏、风险偏好下降但指数未硬破位", "holding": "周五评分，下个交易日买入前 5 名",
        "risk_note": "低波动策略在强牛市可能跑输高弹性品种，也可能集中到传统防御行业。",
        "default_top_n": 5, "tags": ["低波动", "风格轮动", "防守", "大盘过滤"], "market_filter": True, "retention_multiple": 3, "universe_size": 1000,
    },
    "small_large_style_rotation": {
        "name": "小盘/大盘风格轮动代理", "summary": "用成交额代理市值风格：小盘活跃时偏小盘动量，大盘核心走强时偏高流动性核心资产。",
        "formula": "30%小盘代理 + 25%20日动量 + 20%60日趋势 + 15%低波动 + 10%流动性底线",
        "basis": "A 股风格经常在小盘题材和大盘核心之间切换；当前数据没有真实市值字段，因此用成交额反向近似小盘，同时设置流动性底线。",
        "risk": "aggressive", "risk_label": "激进", "market_fit": "小盘风格占优、题材扩散、大盘未处于硬下跌", "holding": "周五评分，下个交易日买入前 4 名",
        "risk_note": "成交额代理市值并不精确，小盘端冲击成本和停牌/ST风险更高，实盘必须降仓位。",
        "default_top_n": 4, "tags": ["风格轮动", "小盘代理", "动量", "大盘过滤"], "market_filter": True, "retention_multiple": 1, "universe_size": 2500,
    },
    "low_vol_high_momentum_social": {
        "name": "低波高动量社媒版", "summary": "筛选动量向上但波动不过高的股票，避免纯追热点。",
        "formula": "35%20日动量 + 25%低波动 + 20%60日趋势 + 10%温和放量 + 10%流动性",
        "basis": "综合 Reddit/X 上常见 momentum + volatility control 思路，以及 A 股 ETF 短期动量研究。",
        "risk": "balanced", "risk_label": "均衡", "market_fit": "热点持续但分化明显的行情", "holding": "周五评分，下个交易日买入前 5 名",
        "risk_note": "动量策略在横盘震荡中容易反复换仓。",
        "default_top_n": 5, "tags": ["动量", "低波动", "量价", "社媒思路"], "market_filter": True, "retention_multiple": 2, "universe_size": 1000,
    },
}


def preset_catalog() -> list[dict]:
    return [{"id": key, **value} for key, value in PRESETS.items()]


for definition in PRESETS.values():
    definition.setdefault("market_filter", True)
    definition.setdefault("retention_multiple", 2)


def _compute_weekly_feature_panel(store: ParquetMarketStore, start: date, end: date) -> pd.DataFrame:
    manifest = store.manifest()
    if manifest.get("status") == "running":
        completed = int(manifest.get("completed_symbols", 0))
        total = int(manifest.get("total_symbols", 0))
        raise RuntimeError(f"A 股行情正在同步（{completed}/{total}），完成质量审计前禁止回测")
    if store.audit_path.exists():
        import json
        audit = json.loads(store.audit_path.read_text(encoding="utf-8"))
        if audit.get("status") not in {"passed", "passed_with_quarantine", "passed_with_warnings"}:
            raise RuntimeError("行情数据审计未通过，已停止回测以避免使用错误复权数据")
    manifest_checksum = str(store.manifest().get("checksum") or "no-checksum")[:16]
    cache_dir = store.root / "feature_cache"
    industry_path = store.root / "industry_snapshot.parquet"
    industry_version = str(industry_path.stat().st_mtime_ns) if industry_path.exists() else "no-industry"
    cache_path = cache_dir / f"weekly_v6_{manifest_checksum}_{industry_version}_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    universe_path = store.root / "universe.parquet"
    names = {}
    industries = {}
    excluded_symbols: set[str] = set()
    if store.audit_path.exists():
        import json
        audit = json.loads(store.audit_path.read_text(encoding="utf-8"))
        excluded_symbols.update(audit.get("quarantine", []))
    if universe_path.exists():
        universe = pd.read_parquet(universe_path)
        names = dict(zip(universe["symbol"], universe["name"], strict=False))
        if not store.timeline_ready():
            excluded_symbols.update(universe.loc[universe["name"].map(is_excluded_security_name), "symbol"])
    if industry_path.exists():
        industry_snapshot = pd.read_parquet(industry_path)
        industries = dict(zip(industry_snapshot["symbol"], industry_snapshot["industry"], strict=False))
    frames: list[pd.DataFrame] = []
    read_start = start - timedelta(days=180)
    for path in sorted(store.bars_dir.glob("*.parquet")):
        if path.stem in excluded_symbols:
            continue
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date")
        frame["listed_days"] = np.arange(1, len(frame) + 1)
        frame = frame[(frame["date"].dt.date >= read_start) & (frame["date"].dt.date <= end)].sort_values("date")
        if len(frame) < 65:
            continue
        close = frame.get("adj_close", frame["close"]).astype(float)
        returns = close.pct_change()
        frame["ret1"] = close.pct_change()
        frame["ret5"] = close.pct_change(5)
        frame["ret10"] = close.pct_change(10)
        frame["ret20"] = close.pct_change(20)
        frame["ret60"] = close.pct_change(60)
        frame["vol20"] = returns.rolling(20).std() * np.sqrt(252)
        frame["downside_vol20"] = returns.where(returns < 0, 0).rolling(20).std() * np.sqrt(252)
        frame["trend60"] = close / close.rolling(60).mean() - 1
        frame["ma20_gap"] = close / close.rolling(20).mean() - 1
        frame["ma5_gap"] = close / close.rolling(5).mean() - 1
        deviation = close.rolling(20).std(ddof=0)
        frame["boll_z"] = (close - close.rolling(20).mean()) / deviation.replace(0, np.nan)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        frame["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        frame["volume_ratio"] = frame["volume"].rolling(5).mean() / frame["volume"].rolling(20).mean().replace(0, np.nan)
        frame["liquidity"] = frame["amount"].rolling(20).mean()
        high60 = frame.get("adj_high", frame["high"]).astype(float).rolling(60).max()
        low60 = frame.get("adj_low", frame["low"]).astype(float).rolling(60).min()
        frame["range60"] = high60 / low60.replace(0, np.nan) - 1
        frame["high60_pos"] = close / high60.replace(0, np.nan)
        frame["next_date"] = frame["date"].shift(-1)
        frame["next_open"] = frame["open"].shift(-1)
        frame["week"] = frame["date"].dt.to_period("W-FRI").astype(str)
        weekly = frame.groupby("week", as_index=False).tail(1).copy()
        weekly = weekly[(weekly["date"].dt.date >= start) & weekly["next_open"].notna() & ~weekly["suspended"]]
        weekly["name"] = weekly["symbol"].map(names).fillna(weekly["symbol"])
        weekly["industry"] = weekly["symbol"].map(industries).fillna("未知")
        frames.append(weekly[["week", "date", "next_date", "symbol", "name", "industry", "close", "next_open", "ret1", "ret5", "ret10", "ret20", "ret60", "vol20", "downside_vol20", "trend60", "ma5_gap", "ma20_gap", "boll_z", "rsi14", "volume_ratio", "liquidity", "range60", "high60_pos", "listed_days"]])
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if panel.empty:
        return panel
    benchmark = store.benchmark(read_start, end)
    if benchmark.empty:
        raise RuntimeError("上证指数基准数据缺失，已停止回测以避免生成全程空仓的假结果")
    benchmark = benchmark.sort_values("date").copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark["market_ma20"] = benchmark["close"].rolling(20).mean()
    benchmark["market_ma60"] = benchmark["close"].rolling(60).mean()
    hard_downtrend = (benchmark["close"] < benchmark["market_ma20"]) & (benchmark["market_ma20"] < benchmark["market_ma60"])
    benchmark["market_risk_on"] = ~hard_downtrend
    signals = benchmark[["date", "market_risk_on"]].rename(columns={"date": "market_date"})
    ordered = panel.sort_values("date")
    ordered = pd.merge_asof(ordered, signals.sort_values("market_date"), left_on="date", right_on="market_date", direction="backward")
    result = ordered.drop(columns=["market_date"])
    csi500 = store.benchmark(read_start, end, "000905.SH")
    if csi500.empty:
        result["csi500_risk_on"] = pd.NA
    else:
        csi500 = csi500.sort_values("date").copy()
        csi500["date"] = pd.to_datetime(csi500["date"])
        csi500["ma20"] = csi500["close"].rolling(20).mean()
        csi500["ma60"] = csi500["close"].rolling(60).mean()
        csi500["csi500_risk_on"] = ~((csi500["close"] < csi500["ma20"]) & (csi500["ma20"] < csi500["ma60"]))
        result = pd.merge_asof(
            result.sort_values("date"),
            csi500[["date", "csi500_risk_on"]].sort_values("date"),
            on="date",
            direction="backward",
        )
    all_a = panel.groupby("week", sort=True)["ret5"].mean().clip(lower=-0.20, upper=0.20).rename("weekly_return").reset_index()
    all_a["equal_weight_close"] = (1 + all_a["weekly_return"].fillna(0)).cumprod()
    all_a["ma4"] = all_a["equal_weight_close"].rolling(4).mean()
    all_a["ma12"] = all_a["equal_weight_close"].rolling(12).mean()
    all_a["all_a_equal_weight_risk_on"] = ~(
        (all_a["equal_weight_close"] < all_a["ma4"]) & (all_a["ma4"] < all_a["ma12"])
    )
    all_a_risk = dict(zip(all_a["week"], all_a["all_a_equal_weight_risk_on"], strict=False))
    result["all_a_equal_weight_risk_on"] = result["week"].map(all_a_risk).fillna(False).astype(bool)
    cross_section = panel.groupby("week", sort=True).agg(
        market_breadth20=("ret20", lambda values: float((values > 0).mean())),
        market_breadth5=("ret5", lambda values: float((values > 0).mean())),
        market_median_vol20=("vol20", "median"),
        market_dispersion_ret5=("ret5", "std"),
        market_median_ret20=("ret20", "median"),
    ).reset_index()
    result = result.merge(cross_section, on="week", how="left")
    result["market_risk_on_v1"] = (
        result["market_risk_on"].fillna(False).astype(bool)
        & (
            result["csi500_risk_on"].fillna(False).astype(bool)
            | result["all_a_equal_weight_risk_on"]
        )
    )
    for column in ("date", "next_date"):
        if column in result:
            result[column] = pd.to_datetime(result[column]).astype("datetime64[ns]")
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp.parquet")
    result.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(cache_path)
    return pd.read_parquet(cache_path)


def build_weekly_feature_panel(store: ParquetMarketStore, start: date, end: date, prefer_materialized: bool = False) -> pd.DataFrame:
    materialized_path = store.root / "weekly_features_2020_2026.parquet"
    if not prefer_materialized or not materialized_path.exists():
        return store.filter_point_in_time_universe(_compute_weekly_feature_panel(store, start, end))

    dates = pd.read_parquet(materialized_path, columns=["date"])["date"]
    cache_start = pd.to_datetime(dates).min().date()
    cache_end = pd.to_datetime(dates).max().date()
    if end < cache_start:
        return store.filter_point_in_time_universe(_compute_weekly_feature_panel(store, start, end))
    base = pd.read_parquet(
        materialized_path,
        filters=[("date", ">=", pd.Timestamp(max(start, cache_start))), ("date", "<=", pd.Timestamp(min(end, cache_end)))],
    )
    if start < cache_start:
        historical_end = min(end, cache_start - timedelta(days=1))
        historical = _compute_weekly_feature_panel(store, start, historical_end)
        base = pd.concat([historical, base], ignore_index=True)
    for column in ("date", "next_date"):
        if column in base:
            base[column] = pd.to_datetime(base[column]).astype("datetime64[ns]")
    for column in ("ret1", "ret10", "ma5_gap", "ret60", "range60", "high60_pos", "downside_vol20"):
        if column not in base:
            base[column] = np.nan
    for column, default in (
        ("market_breadth20", 1.0), ("market_breadth5", 1.0),
        ("market_median_vol20", 0.0), ("market_dispersion_ret5", 0.0),
        ("market_median_ret20", 0.0),
    ):
        if column not in base:
            base[column] = default
    if "industry" not in base:
        base["industry"] = "未知"
    universe_path = store.root / "universe.parquet"
    excluded: set[str] = set()
    if universe_path.exists():
        universe = pd.read_parquet(universe_path)
        if not store.timeline_ready():
            excluded.update(universe.loc[universe["name"].map(is_excluded_security_name), "symbol"])
    if store.audit_path.exists():
        import json
        audit = json.loads(store.audit_path.read_text(encoding="utf-8"))
        excluded.update(audit.get("quarantine", []))
    base = base[~base["symbol"].isin(excluded)]
    if end > cache_end:
        recent = _compute_weekly_feature_panel(store, max(start, cache_end + timedelta(days=1)), end)
        base = pd.concat([base, recent], ignore_index=True)
    if base.empty:
        return base
    benchmark = store.benchmark(start - timedelta(days=180), end).copy()
    if benchmark.empty:
        raise RuntimeError("上证指数基准数据缺失，已停止回测以避免生成全程空仓的假结果")
    benchmark = benchmark.sort_values("date")
    benchmark["date"] = pd.to_datetime(benchmark["date"]).astype("datetime64[ns]")
    benchmark["market_ma20"] = benchmark["close"].rolling(20).mean()
    benchmark["market_ma60"] = benchmark["close"].rolling(60).mean()
    benchmark["market_risk_on"] = ~((benchmark["close"] < benchmark["market_ma20"]) & (benchmark["market_ma20"] < benchmark["market_ma60"]))
    ordered = base.drop(columns=["market_risk_on"], errors="ignore").sort_values("date")
    ordered["date"] = pd.to_datetime(ordered["date"]).astype("datetime64[ns]")
    result = pd.merge_asof(ordered, benchmark[["date", "market_risk_on"]].sort_values("date"), on="date", direction="backward")
    for column in ("date", "next_date"):
        if column in result:
            result[column] = pd.to_datetime(result[column]).astype("datetime64[ns]")
    return store.filter_point_in_time_universe(result)


def _score(group: pd.DataFrame, preset_id: str) -> pd.Series:
    rank = lambda series, ascending=True: series.rank(pct=True, ascending=ascending, method="average")
    if preset_id == "reversal_5d":
        return rank(group["ret5"], ascending=False) * 0.75 + rank(group["vol20"], ascending=False) * 0.25
    if preset_id == "low_volatility":
        return rank(group["vol20"], ascending=False)
    if preset_id == "low_vol_trend":
        return rank(group["trend60"]) * 0.45 + rank(group["ret20"]) * 0.25 + rank(group["vol20"], ascending=False) * 0.30
    if preset_id == "filtered_reversal":
        return rank(group["ret5"], ascending=False) * 0.60 + rank(group["trend60"]) * 0.15 + rank(group["vol20"], ascending=False) * 0.25
    if preset_id == "price_volume_momentum":
        return rank(group["ret20"]) * 0.45 + rank(group["trend60"]) * 0.35 + rank(group["volume_ratio"].clip(upper=3)) * 0.20 + rank(group["vol20"], ascending=False) * 0.20
    if preset_id == "oversold_rebound_aggressive":
        return rank(group["ret5"], ascending=False) * 0.45 + rank(group["boll_z"], ascending=False) * 0.25 + rank(group["rsi14"], ascending=False) * 0.20 + rank(group["vol20"], ascending=False) * 0.10
    if preset_id == "breakout_acceleration":
        return rank(group["ret20"]) * 0.35 + rank(group["trend60"]) * 0.30 + rank(group["volume_ratio"].clip(upper=3)) * 0.25 + rank(group["vol20"]) * 0.10
    if preset_id == "close_strength_continuation":
        return rank(group["ret1"]) * 0.35 + rank(group["ret5"]) * 0.25 + rank(group["volume_ratio"].clip(upper=3)) * 0.25 + rank(group["trend60"]) * 0.15
    if preset_id == "weekly_defensive_reversal":
        return rank(group["ret5"], ascending=False) * 0.40 + rank(group["vol20"], ascending=False) * 0.30 + rank(group["trend60"]) * 0.20 + rank(group["liquidity"]) * 0.10
    if preset_id == "weekly_trend_quality_proxy":
        return rank(group["trend60"]) * 0.35 + rank(group["ret20"]) * 0.30 + rank(group["vol20"], ascending=False) * 0.25 + rank(group["liquidity"]) * 0.10
    if preset_id == "weekly_multi_factor_guarded":
        calm_volume = 1 - (group["volume_ratio"] - 1).abs().rank(pct=True, method="average")
        return rank(group["ret5"], ascending=False) * 0.25 + rank(group["trend60"]) * 0.25 + rank(group["vol20"], ascending=False) * 0.25 + calm_volume * 0.15 + rank(group["liquidity"]) * 0.10
    if preset_id == "chanlun_macd_td_proxy":
        pullback_depth = (-group["ma20_gap"]).clip(lower=-0.08, upper=0.12)
        center_retest = 1 - (group["boll_z"].clip(lower=-2.5, upper=1.0) + 0.8).abs().rank(pct=True, method="average")
        divergence_repair = rank(group["ret5"]) * 0.55 + rank(group["ret20"], ascending=False) * 0.45
        td_exhaustion = rank(group["ret10"], ascending=False)
        trend_guard = rank(group["trend60"]) * 0.7 + rank(group["ma5_gap"]) * 0.3
        moderate_volume = 1 - (group["volume_ratio"].clip(upper=3) - 1.15).abs().rank(pct=True, method="average")
        return (
            rank(pullback_depth) * 0.18
            + center_retest * 0.12
            + divergence_repair * 0.25
            + td_exhaustion * 0.20
            + trend_guard * 0.15
            + moderate_volume * 0.10
        )
    if preset_id == "social_small_cap_quality_momentum":
        return rank(group["liquidity"], ascending=False) * 0.35 + rank(group["ret20"]) * 0.25 + rank(group["vol20"], ascending=False) * 0.20 + rank(group["trend60"]) * 0.10 + rank(group["liquidity"]) * 0.10
    if preset_id == "small_cap_reversal_guarded":
        return rank(group["liquidity"], ascending=False) * 0.35 + rank(group["ret5"], ascending=False) * 0.30 + rank(group["trend60"]) * 0.20 + rank(group["vol20"], ascending=False) * 0.15
    if preset_id == "low_price_breakout_proxy":
        return rank(group["ret20"]) * 0.30 + rank(group["high60_pos"]) * 0.25 + rank(group["volume_ratio"].clip(upper=3)) * 0.20 + rank(group["vol20"], ascending=False) * 0.15 + rank(group["liquidity"]) * 0.10
    if preset_id == "limit_up_continuation_proxy":
        return rank(group["ret1"]) * 0.35 + rank(group["ret5"]) * 0.25 + rank(group["volume_ratio"].clip(upper=4)) * 0.20 + rank(group["trend60"]) * 0.10 + rank(group["vol20"], ascending=False) * 0.10
    if preset_id == "core_asset_momentum_rotation":
        risk_adjusted = group["ret20"] / group["vol20"].replace(0, np.nan)
        return rank(group["ret20"]) * 0.45 + rank(group["trend60"]) * 0.25 + rank(risk_adjusted) * 0.20 + rank(group["liquidity"]) * 0.10
    if preset_id == "dual_momentum_cash_guard":
        return rank(group["ret20"]) * 0.40 + rank(group["trend60"]) * 0.30 + rank(group["vol20"], ascending=False) * 0.20 + rank(group["liquidity"]) * 0.10

    if preset_id == "liquidity_strength_rotation":
        return rank(group["liquidity"]) * 0.35 + rank(group["ret20"]) * 0.30 + rank(group["vol20"], ascending=False) * 0.20 + rank(group["trend60"]) * 0.15
    if preset_id == "liquidity_strength_rotation_guarded":
        return rank(group["trend60"]) * 0.45 + rank(group["ret20"]) * 0.25 + rank(group["vol20"], ascending=False) * 0.20 + rank(group["liquidity"]) * 0.10
    if preset_id == "low_vol_style_rotation":
        moderate_volume = 1 - (group["volume_ratio"].clip(upper=3) - 1.1).abs().rank(pct=True, method="average")
        return rank(group["vol20"], ascending=False) * 0.45 + rank(group["trend60"]) * 0.25 + rank(group["liquidity"]) * 0.20 + moderate_volume * 0.10
    if preset_id == "small_large_style_rotation":
        return rank(group["liquidity"], ascending=False) * 0.30 + rank(group["ret20"]) * 0.25 + rank(group["trend60"]) * 0.20 + rank(group["vol20"], ascending=False) * 0.15 + rank(group["liquidity"]) * 0.10
    if preset_id == "low_vol_high_momentum_social":
        moderate_volume = 1 - (group["volume_ratio"].clip(upper=3) - 1.2).abs().rank(pct=True, method="average")
        return rank(group["ret20"]) * 0.35 + rank(group["vol20"], ascending=False) * 0.25 + rank(group["trend60"]) * 0.20 + moderate_volume * 0.10 + rank(group["liquidity"]) * 0.10
    return (
        rank(group["ret20"]) * 0.4
        + rank(group["trend60"]) * 0.3
        + rank(group["volume_ratio"].clip(upper=3)) * 0.1
        + rank(group["vol20"], ascending=False) * 0.2
    )


def _candidate_filter(group: pd.DataFrame, preset_id: str) -> pd.DataFrame:
    filtered = group.copy()
    if preset_id in {"social_small_cap_quality_momentum", "small_cap_reversal_guarded", "low_price_breakout_proxy", "limit_up_continuation_proxy"}:
        filtered = filtered[(filtered["close"] >= 3) & (filtered["listed_days"] >= 250) & (filtered["liquidity"] > 0)]
    if preset_id == "social_small_cap_quality_momentum":
        return filtered[(filtered["ret20"] > -0.12) & (filtered["trend60"] > -0.18) & (filtered["vol20"] < filtered["vol20"].quantile(0.85))]
    if preset_id == "small_cap_reversal_guarded":
        return filtered[(filtered["ret5"] < -0.015) & (filtered["ret5"] > -0.18) & (filtered["trend60"] > -0.25)]
    if preset_id == "low_price_breakout_proxy":
        return filtered[(filtered["close"] <= 25) & (filtered["ret20"] > 0) & (filtered["range60"] < 0.55) & (filtered["volume_ratio"] > 1.0)]
    if preset_id == "limit_up_continuation_proxy":
        return filtered[(filtered["ret1"] > 0.07) & (filtered["ret1"] < 0.115) & (filtered["ret5"] > 0.03) & (filtered["volume_ratio"] > 1.15) & (filtered["ma20_gap"] > 0)]
    if preset_id in {"core_asset_momentum_rotation", "dual_momentum_cash_guard", "low_vol_high_momentum_social", "liquidity_strength_rotation", "low_vol_style_rotation"}:
        return filtered[(filtered["ret20"] > 0) & (filtered["trend60"] > -0.03)]
    if preset_id == "liquidity_strength_rotation_guarded":
        base = filtered[(filtered["close"] >= 3) & (filtered["listed_days"] >= 250) & (filtered["liquidity"] > 0)]
        return base[
            (base["ret20"] > 0.02)
            & (base["trend60"] > 0)
            & (base["vol20"] < base["vol20"].quantile(0.80))
        ]
    if preset_id == "small_large_style_rotation":
        base = filtered[(filtered["close"] >= 3) & (filtered["listed_days"] >= 250) & (filtered["liquidity"] > 0)]
        return base[(base["ret20"] > -0.08) & (base["trend60"] > -0.15) & (base["vol20"] < base["vol20"].quantile(0.90))]
    return filtered


def run_scored_backtest(panel: pd.DataFrame, score_fn, top_n: int = 10, initial_cash: float = 1_000_000, rules: TradingRules | None = None, progress: Callable[[dict], None] | None = None, market_filter: bool = False, retention_multiple: int = 1, universe_size: int = 800, candidate_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None) -> dict:
    rules = rules or TradingRules(max_positions=top_n, rebalance_days=5)
    cash = initial_cash
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    rebalances: list[dict] = []
    equity_points: list[dict] = []
    closed_pnls: list[float] = []
    turnover = 0.0
    last_prices: dict[str, float] = {}
    grouped = list(panel.groupby("week", sort=True))
    total_weeks = len(grouped)
    for week_index, (week, raw_group) in enumerate(grouped, start=1):
        group = raw_group.dropna(subset=["next_open", "ret5", "ret20", "vol20", "trend60", "liquidity"]).copy()
        group = group[(group["listed_days"] >= 120) & (group["liquidity"] > 0)]
        if group.empty:
            continue
        execution_dates = pd.to_datetime(group["next_date"], errors="coerce").dropna()
        if execution_dates.empty:
            continue
        common_execution_date = execution_dates.value_counts().index[0]
        executable = group[pd.to_datetime(group["next_date"]) == common_execution_date].copy()
        if executable.empty:
            continue
        risk_on = not market_filter or "market_risk_on" not in executable or bool(executable["market_risk_on"].fillna(False).iloc[0])
        signal_date = str(executable["date"].max().date())
        execution_date = str(common_execution_date.date())
        liquid = executable.nlargest(min(universe_size, len(executable)), "liquidity").copy()
        if candidate_filter:
            liquid = candidate_filter(liquid)
        liquid["score"] = score_fn(liquid) if not liquid.empty else pd.Series(dtype=float)
        ranked = liquid.sort_values(["score", "symbol"], ascending=[False, True])
        positions_before = set(positions)
        security_names = {symbol: position["name"] for symbol, position in positions.items()}
        security_names.update(dict(zip(executable["symbol"], executable["name"], strict=False)))
        if risk_on and not ranked.empty:
            entry_symbols = ranked.head(top_n)["symbol"].tolist()
            retention_symbols = set(ranked.head(max(top_n, top_n * retention_multiple))["symbol"])
            target_symbols_list = [symbol for symbol in positions if symbol in retention_symbols]
            target_symbols_list.extend(symbol for symbol in entry_symbols if symbol not in target_symbols_list)
            target_symbols_list = target_symbols_list[:top_n]
            targets = ranked[ranked["symbol"].isin(target_symbols_list)].copy()
        else:
            targets = liquid.iloc[0:0]
        market = executable.set_index("symbol")
        valid_prices = market["next_open"].dropna()
        last_prices.update(dict(zip(valid_prices.index, valid_prices.astype(float), strict=False)))
        target_symbols = set(targets["symbol"])
        for symbol in sorted(set(positions) - target_symbols):
            if symbol not in market.index:
                continue
            position = positions.pop(symbol)
            row = market.loc[symbol]
            price = float(row["next_open"]) * (1 - rules.slippage_rate)
            notional = price * position["quantity"]
            commission, tax = trade_cost("sell", notional, rules)
            pnl = notional - commission - tax - position["total_cost"]
            cash += notional - commission - tax
            turnover += notional
            closed_pnls.append(pnl)
            trades.append({"date": str(row["next_date"].date()), "symbol": symbol, "name": position["name"], "side": "sell", "quantity": position["quantity"], "price": round(price, 3), "cost": round(commission + tax, 2), "pnl": round(pnl, 2), "reason": "大盘转弱空仓" if not risk_on else "跌出保留排名或无合格候选"})
        marked_value = sum(pos["quantity"] * last_prices.get(symbol, 0.0) for symbol, pos in positions.items())
        target_value = (cash + marked_value) / max(top_n, 1)
        for row in targets.itertuples(index=False):
            if row.symbol in positions:
                continue
            price = float(row.next_open) * (1 + rules.slippage_rate)
            quantity = int(min(target_value, cash) / price / rules.lot_size) * rules.lot_size
            if quantity <= 0:
                continue
            notional = price * quantity
            commission, _ = trade_cost("buy", notional, rules)
            if notional + commission > cash:
                continue
            cash -= notional + commission
            turnover += notional
            positions[row.symbol] = {"quantity": quantity, "name": row.name, "total_cost": notional + commission}
            trades.append({"date": str(row.next_date.date()), "symbol": row.symbol, "name": row.name, "side": "buy", "quantity": quantity, "price": round(price, 3), "cost": round(commission, 2), "pnl": None, "reason": "进入本周目标排名"})
        # Signals use the completed Friday bar and execute at the next trading
        # day's open. Marking the post-trade portfolio with Friday's close
        # would move valuation backwards in time and materially distort P&L.
        market_value = sum(pos["quantity"] * last_prices.get(symbol, 0.0) for symbol, pos in positions.items())
        equity_value = round(cash + market_value, 2)
        previous_equity = equity_points[-1]["equity"] if equity_points else initial_cash
        weekly_return = equity_value / previous_equity - 1 if previous_equity else 0.0
        equity_points.append({"date": execution_date, "equity": equity_value, "weekly_return": round(float(weekly_return), 6)})
        positions_after = set(positions)
        blocked_exits = sorted((positions_before - target_symbols) & positions_after)
        factor_columns = ("score", "ret20", "trend60", "vol20", "liquidity")
        top_candidates = []
        for row in ranked.head(max(top_n, 5)).to_dict("records"):
            top_candidates.append({
                "symbol": row["symbol"],
                "name": row.get("name", row["symbol"]),
                **{column: round(float(row[column]), 6) for column in factor_columns if pd.notna(row.get(column))},
            })
        rebalances.append({
            "week": str(week),
            "signal_date": signal_date,
            "execution_date": execution_date,
            "market_risk_on": bool(risk_on),
            "reason": (
                "存在停牌或缺失行情，部分退出受阻" if blocked_exits
                else "正常轮动" if risk_on and not ranked.empty
                else "大盘趋势向下，空仓" if not risk_on
                else "严格过滤后无合格候选，空仓"
            ),
            "candidate_count": int(len(ranked)),
            "entered": sorted(positions_after - positions_before),
            "exited": sorted(positions_before - positions_after),
            "retained": sorted(positions_before & positions_after),
            "blocked_exits": blocked_exits,
            "positions_after": sorted(positions_after),
            "security_names": {symbol: security_names.get(symbol, symbol) for symbol in sorted(positions_before | positions_after)},
            "top_candidates": top_candidates,
            "equity": equity_value,
            "weekly_return": round(float(weekly_return), 6),
        })
        if progress:
            progress({"completed": week_index, "total": total_weeks, "equity": equity_points.copy(), "trades": trades.copy(), "rebalances": rebalances.copy()})
    equity = pd.Series([point["equity"] for point in equity_points], dtype=float)
    equity_with_initial = pd.concat([pd.Series([initial_cash], dtype=float), equity], ignore_index=True)
    returns = equity_with_initial.pct_change().dropna()
    total_return = equity.iloc[-1] / initial_cash - 1 if not equity.empty else 0
    volatility = returns.std() * np.sqrt(52) if len(returns) > 1 else 0
    metrics = {
        "total_return": round(float(total_return), 6),
        "annual_return": round(float((1 + total_return) ** (52 / max(len(returns), 1)) - 1), 6) if total_return > -1 else -1,
        "volatility": round(float(volatility), 6),
        "sharpe": round(float(returns.mean() / returns.std() * np.sqrt(52)), 4) if len(returns) > 1 and returns.std() else 0,
        "max_drawdown": round(float((equity_with_initial / equity_with_initial.cummax() - 1).min()), 6) if not equity.empty else 0,
        "turnover": round(turnover / initial_cash, 4),
    }
    metrics.update({
        "win_rate": round(sum(pnl > 0 for pnl in closed_pnls) / len(closed_pnls), 4) if closed_pnls else 0,
        "closed_trades": len(closed_pnls),
        "profit_loss_ratio": round(float(np.mean([p for p in closed_pnls if p > 0]) / abs(np.mean([p for p in closed_pnls if p < 0]))), 3) if any(p > 0 for p in closed_pnls) and any(p < 0 for p in closed_pnls) else 0,
    })
    return {
        "metrics": metrics,
        "equity": equity_points,
        "trades": trades,
        "rebalances": rebalances,
        "rules": asdict(rules),
        "backtest_assumptions": {
            "engine_version": "weekly-next-open-v2-20260623",
            "signal_frequency": "weekly",
            "signal_timing": "completed weekly bar",
            "execution_timing": "next trading session open",
            "execution_price": "raw open with configured slippage",
            "initial_equity_in_metrics": True,
            "market_filter": market_filter,
            "universe_size": universe_size,
            "top_n": top_n,
            "retention_multiple": retention_multiple,
        },
    }


def run_preset_backtest(panel: pd.DataFrame, preset_id: str, top_n: int = 10, initial_cash: float = 1_000_000, rules: TradingRules | None = None, progress: Callable[[dict], None] | None = None) -> dict:
    if preset_id not in PRESETS:
        raise ValueError("未知研究策略")
    if preset_id in {"liquidity_strength_rotation_v1", "leader_relay_daily", "alpha191_132_173_research", "alpha191_173_174_083_v2", "alpha191_042_061_095_v3"}:
        raise ValueError("该策略使用独立日频撮合器，不能通过旧周频引擎运行")
    definition = PRESETS[preset_id]
    result = run_scored_backtest(
        panel,
        lambda group: _score(group, preset_id),
        top_n,
        initial_cash,
        rules,
        progress,
        bool(definition.get("market_filter")),
        int(definition.get("retention_multiple", 1)),
        int(definition.get("universe_size", 800)),
        lambda group: _candidate_filter(group, preset_id),
    )
    result["preset"] = {"id": preset_id, **PRESETS[preset_id]}
    return result


def chart_payload(store: ParquetMarketStore, symbol: str, start: date, end: date, trades: list[dict]) -> dict:
    from app.services.technical_indicators import indicator_records

    bars = store.read(start, end, [symbol])
    names = pd.read_parquet(store.root / "universe.parquet") if (store.root / "universe.parquet").exists() else pd.DataFrame()
    name_map = dict(zip(names.get("symbol", []), names.get("name", []), strict=False))
    return {"symbol": symbol, "name": name_map.get(symbol, symbol), "bars": indicator_records(bars), "trades": [item for item in trades if item["symbol"] == symbol]}
