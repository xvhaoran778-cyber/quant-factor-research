"""自动交易引擎 - 无人值守全自动量化交易"""

import os, sys, json, time
from typing import Dict, List
from datetime import datetime
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.paper_account import PaperAccount
from autotrader.rules import is_trading_time, can_buy_today, calc_shares, calc_commission
from autotrader.risk import RiskManager
from notify.pushplus import PushPlusNotifier


class AutoTrader:
    """全自动交易引擎"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        at_config = self.config.get('autotrader', {})
        
        # 模拟账户（独立存储路径，不与仪表盘共用）
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                'data', 'autotrader_account.json')
        self.account = PaperAccount(
            initial_capital=at_config.get('initial_capital', 10000),
            save_path=save_path
        )
        
        # 风控
        self.risk = RiskManager(at_config.get('risk', {}))
        
        # 微信通知
        pushplus_token = self.config.get('notification', {}).get('pushplus', {}).get('token', '')
        pushplus_enabled = self.config.get('notification', {}).get('pushplus', {}).get('enabled', False)
        if pushplus_token and pushplus_enabled:
            self.notifier = PushPlusNotifier(pushplus_token)
        else:
            self.notifier = None
        
        # 上次快报时间
        self._last_snapshot = None
        
        # 扫描配置
        self.factor_weights = at_config.get('factor_weights', {
            'alpha_return_20d': 0.4, 'alpha_volume_ratio_5d': 0.3,
            'alpha_price_ma_ratio_20': 0.3
        })
        self.scan_count = at_config.get('scan_count', 200)
        self.top_n = at_config.get('top_n', 10)
        
        # 状态
        self.running = False
        self.last_scan = None
        self.scan_history = []
        self.log: List[str] = []
    
    def _log(self, msg: str):
        t = datetime.now().strftime("%H:%M:%S")
        entry = f"[{t}] {msg}"
        logger.info(entry)
        self.log.append(entry)
        if len(self.log) > 200:
            self.log = self.log[-200:]
    
    def scan_and_trade(self, factor_weights: Dict = None) -> Dict:
        """执行一轮扫描-分析-交易"""
        if factor_weights:
            self.factor_weights = factor_weights
        
        result = {
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'is_trading': is_trading_time(),
            'scanned': 0, 'analyzed': 0, 'bought': 0, 'sold': 0, 'errors': 0
        }
        
        self._log(f"开始扫描周期 (交易时间: {is_trading_time()})")
        
        # 1. 加载股票列表
        cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'data', 'stock_list_cache.json')
        stock_list = []
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                all_stocks = json.load(f)
            stock_list = [c for c in all_stocks.keys() 
                         if c.isdigit() and len(c)==6 
                         and not c.startswith(('688','399','8','4','2','300','301'))
                         and 'ST' not in all_stocks.get(c, '')
                         and c.startswith(('000','001','002','003','600','601','602','603','605'))]
        if not stock_list:
            self._log("无可用股票列表")
            return result
        
        stock_list = stock_list[:self.scan_count]
        result['scanned'] = len(stock_list)
        
        # 2. 因子扫描
        try:
            from factors.fast_scanner import FastScanner
            scanner = FastScanner(max_workers=8)
            scores_df = scanner.scan(stock_list, self.factor_weights,
                                    top_n=self.top_n, min_volume=50000,
                                    use_full_alpha=False)
        except Exception as e:
            self._log(f"扫描失败: {e}")
            result['errors'] += 1
            return result
        
        if scores_df.empty:
            self._log("扫描无结果")
            return result
        
        self._log(f"扫描完成: {len(scores_df)}只候选, Top1: {scores_df.iloc[0].get('name','')}({scores_df.iloc[0]['code']}) 得分{scores_df.iloc[0]['score']:.0f}")
        
        # 3. 持仓巡检（先检查卖出）
        sold_codes = self._check_positions()
        result['sold'] = len(sold_codes)
        
        # 4. 候选筛选（排除已持仓）
        held_codes = set(self.account.positions.keys())
        candidates = scores_df[~scores_df['code'].isin(held_codes)].head(5)
        
        if candidates.empty:
            self._log("暂无新的买入候选")
            return result
        
        # 5. Agent分析+买入
        bought_codes = self._analyze_and_buy(candidates)
        result['bought'] = len(bought_codes)
        result['analyzed'] = len(candidates)
        
        self.last_scan = result
        self.scan_history.append(result)
        if len(self.scan_history) > 500:
            self.scan_history = self.scan_history[-500:]
        
        # 每2小时发送持仓快报
        if self.notifier:
            now = time.time()
            if not self._last_snapshot or (now - self._last_snapshot) > 7200:
                summary = self.account.get_summary()
                pos_text = ""
                for code, pos in self.account.positions.items():
                    pos_text += f"- {pos.name}({code}) {pos.shares}股 @ ¥{pos.current_price:.2f} 盈亏{pos.profit_pct:+.1f}%\n"
                if not pos_text:
                    pos_text = "- 空仓\n"
                self.notifier.send(
                    f"📊 持仓快报 权益¥{summary['total_equity']:,.0f}",
                    f"总权益 ¥{summary['total_equity']:,.0f}\n累计收益 {summary['total_return']:+.2f}% ({len(self.account.positions)}只持仓)\n\n{pos_text}",
                    "markdown"
                )
                self._last_snapshot = now
        
        return result
    
    def _check_positions(self) -> List[str]:
        """持仓巡检：止损/止盈/Agent卖出"""
        if not self.account.positions:
            return []
        
        # 更新持仓价格
        try:
            from data.collectors import TencentCollector
            tc = TencentCollector({})
            tc.connect()
            codes = list(self.account.positions.keys())
            quotes = tc.batch_quotes(codes)
            if quotes:
                quote_dict = {q['code']: q for q in quotes}
                for code, pos in self.account.positions.items():
                    if code in quote_dict:
                        pos.current_price = quote_dict[code].get('price', pos.current_price)
                        pos.market_value = pos.shares * pos.current_price
                        pos.profit = (pos.current_price - pos.avg_cost) * pos.shares
                        pos.profit_pct = (pos.current_price / pos.avg_cost - 1) * 100
        except:
            pass
        
        sold = []
        for code, pos in list(self.account.positions.items()):
            if pos.current_price <= 0:
                continue
            
            # T+1检查
            if not can_buy_today(pos.buy_date):
                continue
            
            # 止损
            if self.risk.should_stop_loss(pos.avg_cost, pos.current_price):
                self.account.sell(code, pos.current_price, reason="止损")
                self._log(f"🔴 止损卖出 {pos.name}({code}) @ {pos.current_price:.2f} 亏损{pos.profit_pct:.1f}%")
                if self.notifier:
                    self.notifier.send("⚠️ 止损触发", f"**{pos.name}**({code})\n止损卖出 @ ¥{pos.current_price:.2f}\n亏损 {pos.profit_pct:.1f}%", "markdown")
                sold.append(code)
                continue
            
            # 止盈
            if self.risk.should_take_profit(pos.avg_cost, pos.current_price):
                self.account.sell(code, pos.current_price, reason="止盈")
                self._log(f"🟢 止盈卖出 {pos.name}({code}) @ {pos.current_price:.2f} 盈利{pos.profit_pct:.1f}%")
                if self.notifier:
                    self.notifier.send("🟢 止盈触发", f"**{pos.name}**({code})\n止盈卖出 @ ¥{pos.current_price:.2f}\n盈利 +{pos.profit_pct:.1f}%", "markdown")
                sold.append(code)
        
        return sold
    
    def _analyze_and_buy(self, candidates) -> List[str]:
        """Agent分析+自动买入"""
        from agents import FundamentalAgent, TechnicalAgent, SentimentAgent, NewsAgent, MacroAgent
        llm_config = self.config.get('llm', {})
        agent_cfg = {'llm': llm_config,
                     'akshare': self.config.get('data_sources', {}).get('akshare', {}),
                     'tencent': self.config.get('data_sources', {}).get('tencent', {})}
        
        bought = []
        
        for _, row in candidates.iterrows():
            code = row['code']
            name = row.get('name', code)
            price = row.get('price', 0)
            
            if price <= 0:
                continue
            if not self.risk.filter_stock(code, price):
                continue
            if not self.risk.check_daily_limit():
                self._log(f"今日交易次数已达上限，停止买入")
                break
            
            # 计算买入股数
            available = min(self.account.capital * 0.5, 2000)
            shares = calc_shares(available, price)
            if shares <= 0:
                continue
            
            # 5Agent并行分析
            self._log(f"Agent分析: {name}({code})...")
            
            agents = {
                '基本面': FundamentalAgent(agent_cfg),
                '技术面': TechnicalAgent(agent_cfg),
                '情绪': SentimentAgent(agent_cfg),
                '消息面': NewsAgent(agent_cfg),
                '宏观': MacroAgent(agent_cfg)
            }
            
            votes = {'强烈买入': 0, '买入': 0, '持有': 0, '卖出': 0, '强烈卖出': 0}
            agent_reasons = {}
            
            def _run(aname, aobj, c):
                try:
                    r = aobj.analyze(c)
                    return aname, r.signal.signal_type.value, r.signal.reason
                except:
                    return aname, '持有', '分析异常'
            
            with ThreadPoolExecutor(max_workers=5) as ex:
                futures = {ex.submit(_run, n, o, code): n for n, o in agents.items()}
                for f in as_completed(futures):
                    try:
                        aname, sig, reason = f.result()
                        votes[sig] = votes.get(sig, 0) + 1
                        agent_reasons[aname] = reason
                    except:
                        pass
            
            buy_votes = votes['强烈买入'] + votes['买入']
            
            if buy_votes >= 3:
                ok = self.account.buy(code, name, price, shares=shares,
                                     reason=f"Agent {buy_votes}/5票→买入")
                if ok:
                    self.risk.record_trade()
                    self._log(f"🟢 买入 {name}({code}) {shares}股 @ {price:.2f} (Agent {buy_votes}/5票)")
                    if self.notifier:
                        self.notifier.send(
                            f"🟢 买入 {name}",
                            f"**{name}**({code})\n买入 {shares}股 @ ¥{price:.2f}\n金额 ¥{shares*price:,.0f}\nAgent {buy_votes}/5票通过",
                            "markdown"
                        )
                    bought.append(code)
            else:
                self._log(f"⏸️ {name}({code}) Agent投票不足({buy_votes}/5)，跳过")
        
        return bought
    
    def get_status(self) -> Dict:
        """获取状态"""
        summary = self.account.get_summary()
        summary['running'] = self.running
        summary['last_scan'] = self.last_scan
        summary['log'] = self.log[-30:]
        return summary
