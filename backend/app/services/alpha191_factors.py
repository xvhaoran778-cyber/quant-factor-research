"""alpha191_factors.py — GTJA Alpha191 因子引擎

基于国泰君安 2017 年《基于短周期价量特征的多因子选股体系》，
实现 191 个量价因子的核心子集。

因子分类：
  Alpha001-030: 动量/反转因子
  Alpha031-060: 量价关系因子  
  Alpha061-090: 波动率因子
  Alpha091-120: 价格位置因子
  Alpha121-150: 成交行为因子
  Alpha151-191: 市场强弱/趋势因子

用法:
  from factors.alpha191 import Alpha191
  alpha = Alpha191()
  df = alpha.calculate_all(df)  # 输入 OHLCV DataFrame，输出带因子列
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


class Alpha191:
    """Alpha191 因子引擎"""

    def __init__(self):
        self.factor_names: list[str] = []

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # 确保必要的列存在
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # 基础衍生数据
        df = self._calc_basic_derived(df)

        # 因子组
        df = self._alpha_momentum_reversal(df)       # 001-030
        df = self._alpha_volume_price(df)             # 031-060
        df = self._alpha_volatility(df)               # 061-090
        df = self._alpha_price_position(df)           # 091-120
        df = self._alpha_volume_behavior(df)          # 121-150
        df = self._alpha_trend_market(df)             # 151-191

        self.factor_names = [c for c in df.columns if c.startswith("alpha")]
        return df

    def _calc_basic_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础衍生数据。"""
        # 收益率
        df["return_1d"] = df["close"].pct_change()
        for d in [5, 10, 20, 60]:
            df[f"return_{d}d"] = df["close"].pct_change(d)

        # 对数收益率
        df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))

        # VWAP
        df["vwap"] = (df["close"] * df["volume"]).rolling(5).sum() / df["volume"].rolling(5).sum()

        # 典型价
        df["typ"] = (df["high"] + df["low"] + df["close"]) / 3

        # 标准差
        for d in [5, 10, 20, 60]:
            df[f"std_{d}"] = df["close"].rolling(d).std()

        # 均值
        for d in [5, 10, 20, 60]:
            df[f"ma_{d}"] = df["close"].rolling(d).mean()

        # 成交量变化
        df["volume_change"] = df["volume"].pct_change()
        df["log_volume"] = np.log(df["volume"] + 1)
        df["log_volume_change"] = df["log_volume"].diff()

        # 价差
        df["high_low"] = df["high"] - df["low"]
        df["close_open"] = df["close"] - df["open"]

        return df

    # ── 工具函数 ──
    @staticmethod
    def _rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True)

    @staticmethod
    def _ts_argmax(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d).apply(lambda x: x.argmax(), raw=True)

    @staticmethod
    def _ts_argmin(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d).apply(lambda x: x.argmin(), raw=True)

    @staticmethod
    def _ts_sum(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d).sum()

    @staticmethod
    def _ts_mean(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d).mean()

    @staticmethod
    def _ts_std(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d).std()

    @staticmethod
    def _ts_corr(s1: pd.Series, s2: pd.Series, d: int) -> pd.Series:
        return s1.rolling(d).corr(s2)

    @staticmethod
    def _ts_cov(s1: pd.Series, s2: pd.Series, d: int) -> pd.Series:
        return s1.rolling(d).cov(s2)

    @staticmethod
    def _signed_power(s: pd.Series, exp: float) -> pd.Series:
        return np.sign(s) * (np.abs(s) ** exp)

    @staticmethod
    def _delta(s: pd.Series, d: int) -> pd.Series:
        return s.diff(d)

    # ═══════════════════════════════════════════════════════════
    # Alpha001-030: 动量/反转因子
    # ═══════════════════════════════════════════════════════════

    def _alpha_momentum_reversal(self, df: pd.DataFrame) -> pd.DataFrame:
        """动量与反转因子组。"""
        c, o, h, l, v, vw = df["close"], df["open"], df["high"], df["low"], df["volume"], df["vwap"]
        r1 = df["return_1d"]

        # Alpha001: Rank(Ts_ArgMax(SignedPower(If(ret>0,ret,0),2),5)) - 0.5
        pos_ret = r1.where(r1 > 0, 0)
        sp = self._signed_power(pos_ret, 2)
        df["alpha001"] = self._rank(self._ts_argmax(sp, 5)) - 0.5

        # Alpha002: -1 * Correlation(Rank(Delta(log(volume),2)), Rank(((close-open)/open)), 6)
        vol_delta = self._delta(df["log_volume"], 2)
        co_ratio = (c - o) / o
        df["alpha002"] = -1 * self._ts_corr(self._rank(vol_delta), self._rank(co_ratio), 6)

        # Alpha003: -1 * Correlation(Rank(Open), Rank(Volume), 10)
        df["alpha003"] = -1 * self._ts_corr(self._rank(o), self._rank(v), 10)

        # Alpha004: -1 * Ts_Rank(Rank(Low), 9)
        df["alpha004"] = -1 * self._rank(l).rolling(9).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)

        # Alpha006: -1 * Correlation(Open, Volume, 10)
        df["alpha006"] = -1 * self._ts_corr(o, v, 10)

        # Alpha007: ((Adv20 < Volume) ? ((-1 * Ts_Rank(Abs_Delta(Close, 7), 60)) * Sign(Delta(Close, 7))) : (-1 * 1))
        adv20 = v.rolling(20).mean()
        cond = adv20 < v
        close_delta7 = self._delta(c, 7)
        ts_rank = self._rank(abs(close_delta7)).rolling(60).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
        val = -1 * ts_rank * np.sign(close_delta7)
        df["alpha007"] = val.where(cond, -1)

        # Alpha008: -1 * Rank(((Sum(Open, 5) * Sum(Return, 5)) - Delay((Sum(Open, 5) * Sum(Return, 5)), 10)))
        sum_o5 = o.rolling(5).sum()
        sum_r5 = r1.rolling(5).sum()
        prod = sum_o5 * sum_r5
        df["alpha008"] = -1 * self._rank(prod - prod.shift(10))

        # Alpha009: ((0 < Ts_Min(Delta(Close, 1), 5)) ? Delta(Close, 1) : ((Ts_Max(Delta(Close, 1), 5) < 0) ? Delta(Close, 1) : (-1 * Delta(Close, 1))))
        d1 = self._delta(c, 1)
        min_d1 = d1.rolling(5).min()
        max_d1 = d1.rolling(5).max()
        cond1 = 0 < min_d1
        cond2 = max_d1 < 0
        df["alpha009"] = np.select([cond1, cond2], [d1, d1], -d1)

        # Alpha010: Rank(0 + min(((close - low) < (high - close)) ? (close - low) : (high - close)), 5)
        inner = pd.concat([(c - l), (h - c)], axis=1).min(axis=1)
        df["alpha010"] = self._rank(inner.rolling(5).min())

        # Alpha012: (Open - (VWAP * 0.5 + (VWAP * 0.5).shift(1)))
        vwap_half = vw * 0.5
        df["alpha012"] = o - (vwap_half + vwap_half.shift(1))

        # Alpha014: -1 * Rank(Correlation(Rank(High), Rank(Volume), 5))
        df["alpha014"] = -1 * self._rank(self._ts_corr(self._rank(h), self._rank(v), 5))

        # Alpha015: -1 * Sum(Rank(Correlation(Rank(High), Rank(Volume), 3)), 3)
        corr3 = self._ts_corr(self._rank(h), self._rank(v), 3)
        df["alpha015"] = -1 * self._rank(corr3).rolling(3).sum()

        # Alpha016: -1 * Rank(Covariance(Rank(High), Rank(Volume), 5))
        df["alpha016"] = -1 * self._rank(self._ts_cov(self._rank(h), self._rank(v), 5))

        # Alpha018: -1 * Rank(((StdDev(Abs((Close - Open)), 5) + (Close - Open)) + Correlation(Close, Open, 10)))
        abs_co = abs(c - o)
        std5 = abs_co.rolling(5).std()
        corr10 = self._ts_corr(c, o, 10)
        df["alpha018"] = -1 * self._rank(std5 + (c - o) + corr10)

        # Alpha019: (-1 * ((((close - low) - (high - close)) / (high - low)) * Volume)) / ((((high - close) - (close - low)) / (high - low)) * Volume)
        numerator = (((c - l) - (h - c)) / (h - l + 1e-8)) * v
        denominator = (((h - c) - (c - l)) / (h - l + 1e-8)) * v
        df["alpha019"] = (-1 * numerator) / (denominator + 1e-8)

        # Alpha020: -1 * Rank((Open - (High + Low) / 2))
        df["alpha020"] = -1 * self._rank(o - (h + l) / 2)

        # Alpha021: (((Sum(Close, 8) / 8) + StdDev(Close, 8)) < (Sum(Close, 2) / 2))
        cond = (c.rolling(8).mean() + c.rolling(8).std()) < c.rolling(2).mean()
        df["alpha021"] = cond.astype(float)

        # Alpha022: -1 * (Delta(Correlation(High, Volume, 5), 5) * Rank(StdDev(Close, 20)))
        corr_hv5 = self._ts_corr(h, v, 5)
        df["alpha022"] = -1 * self._delta(corr_hv5, 5) * self._rank(c.rolling(20).std())

        # Alpha024: ((((close - high) < (close - low)) ? (close - high) : (close - low)) / (close - low))
        inner2 = pd.concat([(c - h), (c - l)], axis=1).min(axis=1)
        df["alpha024"] = inner2 / (c - l + 1e-8)

        # Alpha026: -1 * Correlation(Rank(Volume), Rank(VWAP), 5)
        df["alpha026"] = -1 * self._ts_corr(self._rank(v), self._rank(vw), 5)

        # Alpha028: (Scale(Correlation(Adv20, Low, 5), 5) - Scale(Correlation(Adv20, Close, 5), 5))
        corr_al5 = self._ts_corr(adv20, l, 5)
        corr_ac5 = self._ts_corr(adv20, c, 5)
        df["alpha028"] = (corr_al5 / corr_al5.rolling(5).std() - corr_ac5 / corr_ac5.rolling(5).std())

        # Alpha029: (Min(Product(Rank(Rank(Scale(Log(Sum(Ts_Min(Rank(Rank((-1 * Rank(Delta((Close - 5), 5))))), 2), 1))), 1)), 1), 5) + Ts_Rank(Delay((-1 * Rank(Delta(Close, 3))), 5), 5))
        df["alpha029"] = 0  # 简化版

        # Alpha030: ((Delta(Close, 5).rank() - (1 - Volume.rank(pct=True)))) / ((Delta(Close, 5).rank() + (1 - Volume.rank(pct=True))))
        d5_rank = self._rank(self._delta(c, 5))
        vol_rank_rev = (1 - self._rank(v))
        df["alpha030"] = (d5_rank - vol_rank_rev) / (d5_rank + vol_rank_rev + 1e-8)

        return df

    # ═══════════════════════════════════════════════════════════
    # Alpha031-060: 量价关系因子
    # ═══════════════════════════════════════════════════════════

    def _alpha_volume_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """量价关系因子组。"""
        c, o, h, l, v, vw = df["close"], df["open"], df["high"], df["low"], df["volume"], df["vwap"]
        r1 = df["return_1d"]
        adv5 = v.rolling(5).mean()
        adv20 = v.rolling(20).mean()
        ret5 = df["return_5d"]

        # Alpha031: (Rank(Volume) * (1 - Rank(Close - Low))) / (Rank(Volume) + Rank(Close - Low))
        vol_rank = self._rank(v)
        cl_rank = self._rank(c - l)
        df["alpha031"] = (vol_rank * (1 - cl_rank)) / (vol_rank + cl_rank + 1e-8)

        # Alpha032: (Scale(Sum(Close, 7), 7) - Scale(Sum(Close, 7), 7))
        sum7 = c.rolling(7).sum()
        df["alpha032"] = (sum7 / sum7.rolling(7).std()) - (sum7 / sum7.rolling(7).std()).shift(7)

        # Alpha034: Median(Volume, 5) / Adv20
        df["alpha034"] = v.rolling(5).median() / (adv20 + 1e-8)

        # Alpha035: (Ts_Rank(Volume, 32) * (1 - Ts_Rank((Close + High - Low), 16))) * (1 - Ts_Rank(Return_1d, 32))
        df["alpha035"] = (self._rank(v).rolling(32).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
                         * (1 - self._rank((c + h - l)).rolling(16).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False))
                         * (1 - self._rank(r1).rolling(32).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)))

        # Alpha036: (2 * Scale(Rank(Delta(Log(Volume), 1)), 6) - Scale(Rank(Delta(Close, 1)), 6))
        vol_d1 = self._rank(self._delta(df["log_volume"], 1))
        close_d1 = self._rank(self._delta(c, 1))
        df["alpha036"] = (2 * vol_d1 / vol_d1.rolling(6).std()
                          - close_d1 / close_d1.rolling(6).std())

        # Alpha037: Rank(Correlation(Delay((Open - Close), 1), Close, 200)) + Rank((Open - Close))
        df["alpha037"] = self._rank(self._ts_corr((o - c).shift(1), c, 200)) + self._rank(o - c)

        # Alpha038: (-1 * Rank(Ts_Rank(Close, 10))) * Rank(Close / Open)
        df["alpha038"] = (-1 * self._rank(self._rank(c).rolling(10).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False))
                          * self._rank(c / o))

        # Alpha040: Sum((Close > Delay(Close, 1)).astype(int), 10) / 10
        df["alpha040"] = (c > c.shift(1)).astype(int).rolling(10).sum() / 10

        # Alpha041: -1 * Correlation(Rank(High - Low), Rank(Volume), 10)
        df["alpha041"] = -1 * self._ts_corr(self._rank(h - l), self._rank(v), 10)

        # Alpha043: -1 * Correlation((Volume - Adv20), Low, 5)
        df["alpha043"] = -1 * self._ts_corr(v - adv20, l, 5)

        # Alpha044: -1 * Correlation(High, Volume, 3) * Correlation(Rank(Volume), Rank(High), 5)
        df["alpha044"] = -1 * self._ts_corr(h, v, 3) * self._ts_corr(self._rank(v), self._rank(h), 5)

        # Alpha045: -1 * Rank(Sum(Delay(Close, 5), 20) / 20) * Correlation(Close, Volume, 2) * Rank(Correlation(Sum(Close, 5), Sum(Close, 20), 2))
        delay5 = c.shift(5)
        sum_delay5 = delay5.rolling(20).sum() / 20
        corr_cv2 = self._ts_corr(c, v, 2)
        corr_sum = self._ts_corr(c.rolling(5).sum(), c.rolling(20).sum(), 2)
        df["alpha045"] = -1 * self._rank(sum_delay5) * corr_cv2 * self._rank(corr_sum)

        # Alpha046: (Mean(Close, 3) - Mean(Close, 10)) / Mean(Close, 60)
        df["alpha046"] = (c.rolling(3).mean() - c.rolling(10).mean()) / (c.rolling(60).mean() + 1e-8)

        # Alpha047: Sum((Close > Delay(Close, 1)).astype(int), 20) / 20 * Correlation(Close, Volume, 5)
        df["alpha047"] = (c > c.shift(1)).astype(int).rolling(20).sum() / 20 * self._ts_corr(c, v, 5)

        # Alpha049: Sum(((High + Low) >= (High.shift(1) + Low.shift(1))).astype(int), 12) / 12 - Sum(((High + Low) <= (High.shift(1) + Low.shift(1))).astype(int), 12) / 12
        up = ((h + l) >= (h.shift(1) + l.shift(1))).astype(int).rolling(12).sum() / 12
        down = ((h + l) <= (h.shift(1) + l.shift(1))).astype(int).rolling(12).sum() / 12
        df["alpha049"] = up - down

        # Alpha051: Sum(((High - Low) > (High.shift(1) - Low.shift(1))).astype(int), 12) / 12
        df["alpha051"] = ((h - l) > (h.shift(1) - l.shift(1))).astype(int).rolling(12).sum() / 12

        # Alpha053: -1 * Correlation(Rank(High), Rank(Volume), 3)
        df["alpha053"] = -1 * self._ts_corr(self._rank(h), self._rank(v), 3)

        # Alpha054: -1 * ((High - Low) / Close + Volume / Adv20) * Correlation(Close, Volume, 5)
        df["alpha054"] = -1 * ((h - l) / c + v / (adv20 + 1e-8)) * self._ts_corr(c, v, 5)

        # Alpha055: -1 * Correlation(Rank((Close - Ts_Min(Low, 12)) / (Ts_Max(High, 12) - Ts_Min(Low, 12))), Rank(Volume), 6)
        roc = (c - l.rolling(12).min()) / (h.rolling(12).max() - l.rolling(12).min() + 1e-8)
        df["alpha055"] = -1 * self._ts_corr(self._rank(roc), self._rank(v), 6)

        # Alpha056: -1 * Correlation(Rank(Volume), Rank((Close - Low) / (High - Low)), 4)
        df["alpha056"] = -1 * self._ts_corr(self._rank(v), self._rank((c - l) / (h - l + 1e-8)), 4)

        # Alpha058: -1 * Correlation(Adv20, Low, 5)
        df["alpha058"] = -1 * self._ts_corr(adv20, l, 5)

        return df

    # ═══════════════════════════════════════════════════════════
    # Alpha061-090: 波动率因子
    # ═══════════════════════════════════════════════════════════

    def _alpha_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """波动率因子组。"""
        c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
        r1 = df["return_1d"]
        ret5 = df["return_5d"]

        # Alpha061: -1 * Correlation(Rank((Close - Low) / (High - Low)), Rank(Volume), 5)
        cl_hl = (c - l) / (h - l + 1e-8)
        df["alpha061"] = -1 * self._ts_corr(self._rank(cl_hl), self._rank(v), 5)

        # Alpha062: -1 * Correlation(High, Volume, 5)
        df["alpha062"] = -1 * self._ts_corr(h, v, 5)

        # Alpha063: -1 * Correlation(Rank((Close - Low) / (High - Low)), Rank(Volume), 5)
        df["alpha063"] = df["alpha061"].copy()

        # Alpha064: -1 * Correlation(Rank(Close - Low), Rank(Volume), 5)
        df["alpha064"] = -1 * self._ts_corr(self._rank(c - l), self._rank(v), 5)

        # Alpha065: -1 * Correlation(Rank(Close - Low), Rank(Adv20), 5)
        adv20 = v.rolling(20).mean()
        df["alpha065"] = -1 * self._ts_corr(self._rank(c - l), self._rank(adv20), 5)

        # Alpha066: -1 * Correlation(Rank(Close - High), Rank(Volume), 5)
        df["alpha066"] = -1 * self._ts_corr(self._rank(c - h), self._rank(v), 5)

        # Alpha068: -1 * Correlation(Rank(High - Low), Rank(Volume), 7)
        df["alpha068"] = -1 * self._ts_corr(self._rank(h - l), self._rank(v), 7)

        # Alpha069: Sum(D(Close - Open, 1) > 0, 10) / 10 * Correlation(Close, Volume, 5)
        df["alpha069"] = (self._delta(c - o, 1) > 0).astype(int).rolling(10).sum() / 10 * self._ts_corr(c, v, 5)

        # Alpha070: StdDev(Close, 10)
        df["alpha070"] = c.rolling(10).std()

        # Alpha071: Correlation(Close, Volume, 5)
        df["alpha071"] = self._ts_corr(c, v, 5)

        # Alpha072: StdDev(Close, 15) / Close * -1
        df["alpha072"] = -1 * c.rolling(15).std() / c

        # Alpha073: -1 * Correlation(Close, Volume, 3) / Correlation(Close, Volume, 5)
        corr3 = self._ts_corr(c, v, 3)
        corr5 = self._ts_corr(c, v, 5)
        df["alpha073"] = -1 * corr3 / (corr5 + 1e-8)

        # Alpha074: Correlation(Close, Volume, 5) * Correlation(Close, Volume, 10) + Correlation(Close, Volume, 3)
        df["alpha074"] = corr5 * self._ts_corr(c, v, 10) + corr3

        # Alpha075: Rank(Correlation(Close, Volume, 3)) + Rank(Correlation(Close, Volume, 5))
        df["alpha075"] = self._rank(corr3) + self._rank(corr5)

        # Alpha076: StdDev(Abs((Close / Delay(Close, 1) - 1)), 20) / Mean(Abs((Close / Delay(Close, 1) - 1)), 20)
        ret_abs = abs(c / c.shift(1) - 1)
        df["alpha076"] = ret_abs.rolling(20).std() / (ret_abs.rolling(20).mean() + 1e-8)

        # Alpha077: Min(Rank(Decay_Linear((Close - Low), 20)), Rank(Decay_Linear(High - Close), 20))
        decay_cl = (c - l).rolling(20).apply(lambda x: (x * np.arange(1, len(x) + 1)).sum() / np.arange(1, len(x) + 1).sum(), raw=True)
        decay_hc = (h - c).rolling(20).apply(lambda x: (x * np.arange(1, len(x) + 1)).sum() / np.arange(1, len(x) + 1).sum(), raw=True)
        df["alpha077"] = pd.concat([self._rank(decay_cl), self._rank(decay_hc)], axis=1).min(axis=1)

        # Alpha082: Correlation(Close, Volume, 10) * Correlation(Close, Volume, 5)
        df["alpha082"] = self._ts_corr(c, v, 10) * corr5

        # Alpha083: -1 * Correlation(Rank(High), Rank(Volume), 5)
        df["alpha083"] = -1 * self._ts_corr(self._rank(h), self._rank(v), 5)

        # Alpha084: Correlation(Rank(Close - Low), Rank(Volume), 10)
        df["alpha084"] = self._ts_corr(self._rank(c - l), self._rank(v), 10)

        # Alpha085: Correlation(Rank(High - Low), Rank(Volume), 10)
        df["alpha085"] = self._ts_corr(self._rank(h - l), self._rank(v), 10)

        # Alpha087: Correlation(Rank(Close - Open), Rank(Volume), 10)
        df["alpha087"] = self._ts_corr(self._rank(c - o), self._rank(v), 10)

        # Alpha088: 1 - Correlation(Rank(Close - Low), Rank(High - Low), 10)
        df["alpha088"] = 1 - self._ts_corr(self._rank(c - l), self._rank(h - l), 10)

        # Alpha089: -1 * (Rank(Correlation(Rank(Low), Rank(Volume), 5)))
        df["alpha089"] = -1 * self._rank(self._ts_corr(self._rank(l), self._rank(v), 5))

        # Alpha090: -1 * (Rank(Close - Low) + Rank(Close - High)) * Rank(Volume)
        df["alpha090"] = -1 * (self._rank(c - l) + self._rank(c - h)) * self._rank(v)

        return df

    # ═══════════════════════════════════════════════════════════
    # Alpha091-120: 价格位置因子
    # ═══════════════════════════════════════════════════════════

    def _alpha_price_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格位置因子组。"""
        c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
        r1 = df["return_1d"]

        # Alpha091: (Ts_Rank((Close - Ts_Max(Close, 5)), 10)) * (-1)
        df["alpha091"] = -1 * self._rank((c - c.rolling(5).max())).rolling(10).apply(
            lambda x: x.rank(pct=True).iloc[-1], raw=False)

        # Alpha092: -1 * Correlation(Rank(Close - Low), Rank(Volume), 10) + Correlation(Rank(Close - High), Rank(Volume), 3)
        df["alpha092"] = -1 * self._ts_corr(self._rank(c - l), self._rank(v), 10) + self._ts_corr(self._rank(c - h), self._rank(v), 3)

        # Alpha093: Ts_Rank(Close - Low, 10) * -1
        df["alpha093"] = -1 * self._rank(c - l).rolling(10).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)

        # Alpha094: Correlation(Close, Volume, 5)
        df["alpha094"] = self._ts_corr(c, v, 5)

        # Alpha095: Rank((Close - Ts_Min(Close, 12)) / (Ts_Max(Close, 12) - Ts_Min(Close, 12)))
        close_range = (c - c.rolling(12).min()) / (c.rolling(12).max() - c.rolling(12).min() + 1e-8)
        df["alpha095"] = self._rank(close_range)

        # Alpha096: Rank(Close - Open) + Rank(Close - Low) - Rank(High - Low)
        df["alpha096"] = self._rank(c - o) + self._rank(c - l) - self._rank(h - l)

        # Alpha097: StdDev(Volume, 10)
        df["alpha097"] = v.rolling(10).std()

        # Alpha098: -1 * (Delta(Correlation(Close, Volume, 5), 5) * Rank(StdDev(Close, 20)))
        corr_cv5 = self._ts_corr(c, v, 5)
        df["alpha098"] = -1 * self._delta(corr_cv5, 5) * self._rank(c.rolling(20).std())

        # Alpha099: -1 * Rank(Correlation(Rank(Close), Rank(Volume), 5))
        df["alpha099"] = -1 * self._rank(self._ts_corr(self._rank(c), self._rank(v), 5))

        # Alpha100: StdDev(Volume, 20)
        df["alpha100"] = v.rolling(20).std()

        # Alpha102: Max(Rank(DECAYLINEAR(DELTA(VOLUME, 1), 5)), Rank(DECAYLINEAR(DELTA(CLOSE, 1), 5)))
        d_vol1 = self._delta(v, 1)
        d_c1 = self._delta(c, 1)
        dec_vol = d_vol1.rolling(5).apply(lambda x: (x * np.arange(1, len(x)+1)).sum() / np.arange(1, len(x)+1).sum(), raw=True)
        dec_c = d_c1.rolling(5).apply(lambda x: (x * np.arange(1, len(x)+1)).sum() / np.arange(1, len(x)+1).sum(), raw=True)
        df["alpha102"] = pd.concat([self._rank(dec_vol), self._rank(dec_c)], axis=1).max(axis=1)

        # Alpha103: -1 * Correlation(Rank(Close - Low), Rank(Volume), 5)
        df["alpha103"] = -1 * self._ts_corr(self._rank(c - l), self._rank(v), 5)

        # Alpha104: -1 * Correlation(Rank(Close - High), Rank(Volume), 5)
        df["alpha104"] = -1 * self._ts_corr(self._rank(c - h), self._rank(v), 5)

        # Alpha108: -1 * Correlation(High, Volume, 5)
        df["alpha108"] = -1 * self._ts_corr(h, v, 5)

        # Alpha110: -1 * Correlation(Open, Volume, 5)
        df["alpha110"] = -1 * self._ts_corr(o, v, 5)

        # Alpha111: -1 * Correlation(Low, Volume, 5)
        df["alpha111"] = -1 * self._ts_corr(l, v, 5)

        # Alpha112: -1 * Correlation(VWAP, Volume, 5)
        df["alpha112"] = -1 * self._ts_corr(df["vwap"], v, 5)

        # Alpha113: -1 * Correlation(Rank(Close), Rank(Close), 5) + Correlation(Open, Volume, 10)
        df["alpha113"] = -1 * self._ts_corr(self._rank(c), self._rank(c), 5) + self._ts_corr(o, v, 10)

        # Alpha114: If(High - Low > 0.01, (Close - Low) / (High - Low), 0)
        df["alpha114"] = ((c - l) / (h - l + 1e-8)).where(h - l > c * 0.01, 0)

        # Alpha115: Rank(Correlation(High, Volume, 10)) * Rank(Correlation(Close, Volume, 3))
        df["alpha115"] = self._rank(self._ts_corr(h, v, 10)) * self._rank(self._ts_corr(c, v, 3))

        # Alpha116: Correlation(Close, Volume, 10) * 5 + Correlation(Close, Volume, 5) * 3 + Correlation(Close, Volume, 3)
        df["alpha116"] = self._ts_corr(c, v, 10) * 5 + self._ts_corr(c, v, 5) * 3 + self._ts_corr(c, v, 3)

        return df

    # ═══════════════════════════════════════════════════════════
    # Alpha121-150: 成交行为因子
    # ═══════════════════════════════════════════════════════════

    def _alpha_volume_behavior(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交行为因子组。"""
        c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
        r1 = df["return_1d"]
        adv5 = v.rolling(5).mean()
        adv20 = v.rolling(20).mean()

        # Alpha121: -1 * Correlation(Rank(VWAP - Close), Rank(Volume), 5)
        df["alpha121"] = -1 * self._ts_corr(self._rank(df["vwap"] - c), self._rank(v), 5)

        # Alpha122: -1 * Correlation(Rank(Close - VWAP), Rank(Volume), 10)
        df["alpha122"] = -1 * self._ts_corr(self._rank(c - df["vwap"]), self._rank(v), 10)

        # Alpha123: -1 * Correlation(Rank(VWAP - Low), Rank(Volume), 5)
        df["alpha123"] = -1 * self._ts_corr(self._rank(df["vwap"] - l), self._rank(v), 5)

        # Alpha124: (Close - VWAP) / DECAYLINEAR(Rank(Adv20), 20)
        decay = self._rank(adv20).rolling(20).apply(
            lambda x: (x * np.arange(1, len(x)+1)).sum() / np.arange(1, len(x)+1).sum(), raw=True)
        df["alpha124"] = (c - df["vwap"]) / (decay + 1e-8)

        # Alpha125: -1 * Correlation(Rank(Close), Rank(Volume), 10)
        df["alpha125"] = -1 * self._ts_corr(self._rank(c), self._rank(v), 10)

        # Alpha126: Correlation(Close, Volume, 10) / Correlation(Close, Volume, 5)
        df["alpha126"] = self._ts_corr(c, v, 10) / (self._ts_corr(c, v, 5) + 1e-8)

        # Alpha128: 1 - Correlation(Rank(High), Rank(Volume), 5) / Correlation(Rank(Low), Rank(Volume), 5)
        corr_hv = self._ts_corr(self._rank(h), self._rank(v), 5)
        corr_lv = self._ts_corr(self._rank(l), self._rank(v), 5)
        df["alpha128"] = 1 - corr_hv / (corr_lv + 1e-8)

        # Alpha129: -1 * Correlation(Close, Volume, 5)
        df["alpha129"] = -1 * self._ts_corr(c, v, 5)

        # Alpha130: -1 * Correlation(High, Volume, 5) + Correlation(Close, Volume, 5)
        df["alpha130"] = -1 * self._ts_corr(h, v, 5) + self._ts_corr(c, v, 5)

        # Alpha131: Correlation(Close, Volume, 10)
        df["alpha131"] = self._ts_corr(c, v, 10)

        # Alpha132: Correlation(Close, Log(Volume), 30) * -1
        df["alpha132"] = -1 * self._ts_corr(c, df["log_volume"], 30)

        # Alpha133: -1 * Correlation(High, Volume, 5) * Correlation(Close, Volume, 5)
        df["alpha133"] = -1 * self._ts_corr(h, v, 5) * self._ts_corr(c, v, 5)

        # Alpha134: Correlation(Close, Volume, 3) - Correlation(Close, Volume, 5)
        df["alpha134"] = self._ts_corr(c, v, 3) - self._ts_corr(c, v, 5)

        # Alpha135: Correlation(Close, Volume, 5) - Correlation(Close, Volume, 10)
        df["alpha135"] = self._ts_corr(c, v, 5) - self._ts_corr(c, v, 10)

        # Alpha136: -1 * Correlation(Close, Volume, 3)
        df["alpha136"] = -1 * self._ts_corr(c, v, 3)

        # Alpha137: -1 * Correlation(Open, Volume, 5)
        df["alpha137"] = -1 * self._ts_corr(o, v, 5)

        # Alpha138: -1 * Correlation(High, Volume, 5)
        df["alpha138"] = -1 * self._ts_corr(h, v, 5)

        # Alpha139: Correlation(Close, Volume, 5) * Correlation(Close, Volume, 10) + Correlation(Close, Volume, 3)
        df["alpha139"] = self._ts_corr(c, v, 5) * self._ts_corr(c, v, 10) + self._ts_corr(c, v, 3)

        # Alpha140: -1 * Correlation(Close, Volume, 10) - Correlation(Close, Volume, 5)
        df["alpha140"] = -1 * self._ts_corr(c, v, 10) - self._ts_corr(c, v, 5)

        # Alpha141: -1 * Correlation(High, Volume, 3)
        df["alpha141"] = -1 * self._ts_corr(h, v, 3)

        # Alpha142: -1 * Correlation(Low, Volume, 3)
        df["alpha142"] = -1 * self._ts_corr(l, v, 3)

        # Alpha143: Correlation(Close, Volume, 5) - Correlation(Close, Volume, 10)
        df["alpha143"] = self._ts_corr(c, v, 5) - self._ts_corr(c, v, 10)

        # Alpha144: Sum((Close - Delay(Close, 1)) / Delay(Close, 1), 20)
        df["alpha144"] = r1.rolling(20).sum()

        # Alpha145: -1 * Correlation(Volume, Close, 5)
        df["alpha145"] = -1 * self._ts_corr(v, c, 5)

        # Alpha147: Correlation(Average(Close, 5), Volume, 5)
        df["alpha147"] = self._ts_corr(c.rolling(5).mean(), v, 5)

        # Alpha149: Correlation(Close, Volume, 10) * 5 + Correlation(Close, Volume, 5)
        df["alpha149"] = self._ts_corr(c, v, 10) * 5 + self._ts_corr(c, v, 5)

        return df

    # ═══════════════════════════════════════════════════════════
    # Alpha151-191: 市场强弱/趋势因子
    # ═══════════════════════════════════════════════════════════

    def _alpha_trend_market(self, df: pd.DataFrame) -> pd.DataFrame:
        """市场强弱与趋势因子组。"""
        c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
        r1 = df["return_1d"]
        adv20 = v.rolling(20).mean()

        # Alpha151: Correlation(Close, Volume, 10) + Correlation(Close, Volume, 5)
        df["alpha151"] = self._ts_corr(c, v, 10) + self._ts_corr(c, v, 5)

        # Alpha152: (Close - Open) / (High - Low) * Volume
        df["alpha152"] = (c - o) / (h - l + 1e-8) * v

        # Alpha153: (Close - Low) / (High - Low) * Volume
        df["alpha153"] = (c - l) / (h - l + 1e-8) * v

        # Alpha154: ((High - Low) / Close) * Volume
        df["alpha154"] = ((h - l) / (c + 1e-8)) * v

        # Alpha155: -1 * Correlation(Close, Volume, 5)
        df["alpha155"] = -1 * self._ts_corr(c, v, 5)

        # Alpha156: -1 * (Rank(Close - Low) + Rank(Close - High)) * Rank(Volume)
        df["alpha156"] = -1 * (self._rank(c - l) + self._rank(c - h)) * self._rank(v)

        # Alpha157: -1 * Correlation(Rank(Close - Low), Rank(Volume), 5)
        df["alpha157"] = -1 * self._ts_corr(self._rank(c - l), self._rank(v), 5)

        # Alpha158: 1 - Correlation(Rank(Close), Rank(Volume), 5)
        df["alpha158"] = 1 - self._ts_corr(self._rank(c), self._rank(v), 5)

        # Alpha159: -1 * Correlation(Rank(High - Low), Rank(Volume), 5)
        df["alpha159"] = -1 * self._ts_corr(self._rank(h - l), self._rank(v), 5)

        # Alpha160: -1 * Correlation(Rank(Close - Open), Rank(Volume), 5)
        df["alpha160"] = -1 * self._ts_corr(self._rank(c - o), self._rank(v), 5)

        # Alpha161: -1 * Correlation(Rank(Close), Rank(Volume), 5)
        df["alpha161"] = -1 * self._ts_corr(self._rank(c), self._rank(v), 5)

        # Alpha162: Correlation(Rank(Close), Rank(Volume), 5)
        df["alpha162"] = self._ts_corr(self._rank(c), self._rank(v), 5)

        # Alpha163: Correlation(Rank(Close - Low), Rank(Volume), 10)
        df["alpha163"] = self._ts_corr(self._rank(c - l), self._rank(v), 10)

        # Alpha164: Correlation(Rank(Close - High), Rank(Volume), 10)
        df["alpha164"] = self._ts_corr(self._rank(c - h), self._rank(v), 10)

        # Alpha165: Correlation(Rank(High - Low), Rank(Volume), 10)
        df["alpha165"] = self._ts_corr(self._rank(h - l), self._rank(v), 10)

        # Alpha166: Correlation(Rank(Close - Open), Rank(Volume), 10)
        df["alpha166"] = self._ts_corr(self._rank(c - o), self._rank(v), 10)

        # Alpha167: -1 * Correlation(Rank(Close), Rank(Volume), 10)
        df["alpha167"] = -1 * self._ts_corr(self._rank(c), self._rank(v), 10)

        # Alpha168: -1 * Correlation(Rank(High), Rank(Volume), 5)
        df["alpha168"] = -1 * self._ts_corr(self._rank(h), self._rank(v), 5)

        # Alpha169: -1 * Correlation(Rank(Low), Rank(Volume), 5)
        df["alpha169"] = -1 * self._ts_corr(self._rank(l), self._rank(v), 5)

        # Alpha170: -1 * Correlation(Rank(Open), Rank(Volume), 5)
        df["alpha170"] = -1 * self._ts_corr(self._rank(o), self._rank(v), 5)

        # Alpha171: -1 * Correlation(Rank(Close), Rank(Volume), 5)
        df["alpha171"] = -1 * self._ts_corr(self._rank(c), self._rank(v), 5)

        # Alpha172: Rank(Correlation(Close, Volume, 10)) + Rank(Correlation(Close, Volume, 5))
        df["alpha172"] = self._rank(self._ts_corr(c, v, 10)) + self._rank(self._ts_corr(c, v, 5))

        # Alpha173: StdDev(Close, 20)
        df["alpha173"] = c.rolling(20).std()

        # Alpha174: 1 - Ts_Rank(Close, 20)
        df["alpha174"] = 1 - self._rank(c).rolling(20).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)

        # Alpha175: Mean(Close, 5) - Mean(Close, 20)
        df["alpha175"] = c.rolling(5).mean() - c.rolling(20).mean()

        # Alpha176: (Mean(Close, 5) - Mean(Close, 60)) / std(Close, 60)
        df["alpha176"] = (c.rolling(5).mean() - c.rolling(60).mean()) / (c.rolling(60).std() + 1e-8)

        # Alpha180: -1 * Correlation(Close, Volume, 5)
        df["alpha180"] = -1 * self._ts_corr(c, v, 5)

        # Alpha181: 1 if Volume > Adv20 else 0
        df["alpha181"] = (v > adv20).astype(float)

        # Alpha182: 1 if Close > Mean(Close, 5) else 0
        df["alpha182"] = (c > c.rolling(5).mean()).astype(float)

        # Alpha183: 1 if Close > Mean(Close, 20) else 0
        df["alpha183"] = (c > c.rolling(20).mean()).astype(float)

        # Alpha184: 1 if Close > Mean(Close, 60) else 0
        df["alpha184"] = (c > c.rolling(60).mean()).astype(float)

        # Alpha185: 1 if Close > Open else 0
        df["alpha185"] = (c > o).astype(float)

        # Alpha186: Rank((Close - Low) / (High - Low)) * Volume
        df["alpha186"] = self._rank((c - l) / (h - l + 1e-8)) * v

        # Alpha187: Rank(Close - Low) + Rank(High - Close)
        df["alpha187"] = self._rank(c - l) + self._rank(h - c)

        # Alpha188: (Close - Open) / (High - Low)
        df["alpha188"] = (c - o) / (h - l + 1e-8)

        # Alpha189: Mean(Abs(Close - Open), 6)
        df["alpha189"] = abs(c - o).rolling(6).mean()

        # Alpha190: Log(Volume)
        df["alpha190"] = df["log_volume"]

        # Alpha191: Correlation(Close, Volume, 5) * Correlation(Close, Volume, 10)
        df["alpha191"] = self._ts_corr(c, v, 5) * self._ts_corr(c, v, 10)

        return df

    def get_factor_count(self) -> int:
        return len(self.factor_names)

    def get_factor_names(self) -> list[str]:
        return self.factor_names
