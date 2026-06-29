"""自动交易独立进程 - 24小时无人值守运行"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, time, yaml
from datetime import datetime
from loguru import logger

from autotrader.engine import AutoTrader
from autotrader.rules import is_trading_time

PID_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'autotrader.pid')
STATUS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'autotrader_status.json')
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'settings.yaml')


def _write_status(data: dict):
    try:
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        with open(STATUS_FILE, 'w') as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    except:
        pass


def _read_status() -> dict:
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}


def main():
    # 写PID
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    
    logger.add("logs/autotrader.log", rotation="10 MB", retention="7 days")
    
    # 加载配置
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    engine = AutoTrader(config)
    engine.running = True
    
    interval = config.get('autotrader', {}).get('interval', 1800)
    
    logger.info(f"自动交易进程启动 PID={os.getpid()}, 间隔={interval}秒")
    logger.info(f"初始资金: ¥{engine.account.initial_capital:,.0f}")
    
    _write_status({
        'pid': os.getpid(),
        'started_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'running',
        'equity': engine.account.get_total_equity(),
        'return': engine.account.get_total_return(),
        'positions': len(engine.account.positions),
        'log': ["进程已启动"],
        'last_scan': None,
        'trades_today': 0
    })
    
    while True:
        try:
            now = datetime.now()
            
            if is_trading_time():
                logger.info(f"交易时段，执行扫描...")
                result = engine.scan_and_trade()
                
                if result:
                    summary = engine.account.get_summary()
                    _write_status({
                        'pid': os.getpid(),
                        'last_run': now.strftime("%H:%M:%S"),
                        'status': 'running',
                        'equity': summary['total_equity'],
                        'return': summary['total_return'],
                        'positions': len(engine.account.positions),
                        'log': engine.log[-30:],
                        'last_scan': result,
                        'trades_today': len(engine.account.trade_history)
                    })
            else:
                # 非交易时段只记录不扫描
                logger.debug(f"非交易时段，休眠中...")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            logger.info("收到停止信号")
            _write_status({**_read_status(), 'status': 'stopped', 'log': ['进程手动停止']})
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
            break
        except Exception as e:
            logger.error(f"扫描异常: {e}")
            _write_status({**_read_status(), 'status': 'error', 'last_error': str(e)})
            time.sleep(60)


if __name__ == '__main__':
    main()
