"""自动交易调度器"""

import sys, os, time, yaml, threading
from typing import Dict
from datetime import datetime
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AutoTraderScheduler:
    """自动交易调度器"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'config', 'settings.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.interval = self.config.get('autotrader', {}).get('interval', 1800)  # 30分钟
        self.engine = None
        self.thread = None
        self.running = False
    
    def start(self):
        """启动后台线程"""
        if self.running:
            return
        
        from autotrader.engine import AutoTrader
        self.engine = AutoTrader(self.config)
        self.engine.running = True
        self.running = True
        
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info(f"自动交易启动，间隔{self.interval}秒")
    
    def stop(self):
        """停止"""
        self.running = False
        if self.engine:
            self.engine.running = False
        logger.info("自动交易已停止")
    
    def _run_loop(self):
        """后台运行循环"""
        while self.running:
            try:
                from autotrader.rules import is_trading_time
                now = datetime.now()
                
                # 仅交易时段扫描（9:30-11:30, 13:00-15:00 周一至五）
                if is_trading_time():
                    self.engine._log(f"交易时段，执行扫描...")
                    self.engine.scan_and_trade()
                else:
                    dormir = self.interval // 60
                    self.engine._log(f"非交易时段，休眠{dormir}分钟 [{now.strftime('%H:%M')}]")
                
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"调度循环异常: {e}")
                time.sleep(60)
    
    def trigger_manual(self) -> Dict:
        """手动触发一次扫描"""
        if self.engine:
            return self.engine.scan_and_trade()
        return {'error': '引擎未初始化'}
    
    def get_status(self) -> Dict:
        """获取状态"""
        if self.engine:
            return self.engine.get_status()
        return {'running': False}


# 独立运行入口
if __name__ == '__main__':
    logger.add("logs/autotrader.log", rotation="1 day")
    scheduler = AutoTraderScheduler()
    scheduler.start()
    
    try:
        print("自动交易已启动，Ctrl+C退出")
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        scheduler.stop()
