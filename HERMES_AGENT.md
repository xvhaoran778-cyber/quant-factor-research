# HermesAgent 操作说明

本项目是本地优先的量化研究系统。HermesAgent 只能在本地做研究、修复、因子开发和回测，不自动上线实盘。

## 当前边界

- 本地项目目录：`/Users/xuhaoran/Documents/agent`
- 本地股票数据：`/Volumes/xhrrrrr_macmini副盘/quantlab/market`
- 前端地址：`http://127.0.0.1:3000`
- 后端地址：`http://127.0.0.1:8000`
- 本地账号：`admin / quantlab`
- 服务器已停止运行 QuantLab；不要自动恢复服务器。

## 启动本地系统

后端：

```bash
cd /Users/xuhaoran/Documents/agent/backend
DATABASE_URL=sqlite:///./quantlab-local.db \
LOCAL_TASK_EAGER=true \
MARKET_DATA_DIR="/Volumes/xhrrrrr_macmini副盘/quantlab/market" \
CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000" \
ADMIN_USERNAME=admin \
ADMIN_PASSWORD=quantlab \
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

前端：

```bash
cd /Users/xuhaoran/Documents/agent/frontend
NEXT_PUBLIC_API_URL=http://localhost:8000/api npm run dev -- --hostname 127.0.0.1 --port 3000
```

## 可以自动做的事

- 修复本地前端或后端 bug。
- 新增因子、策略版本、研究脚本。
- 跑本地回测。
- 生成 `reports/` 下的研究报告。
- 对 V8、V9、MR、Liquidity 等策略做归因和压力测试。
- 读取本地行情 parquet、已有回测 CSV/JSON/Markdown。

## 不允许自动做的事

- 不要上线实盘。
- 不要连接券商真实交易。
- 不要删除股票数据目录。
- 不要删除数据库、Docker volume 或服务器文件。
- 不要重启云服务器服务。
- 不要把 V4 或 V9 作为上线策略。
- 不要基于 proxy 财报因子做上线结论。
- 不要把 JoinQuant proxy 结果当成真实 alpha132 结论。

## 策略事实

- V8 是当前本地核心研究基准：`alpha191_v8_light`。
- V8 高收益但高风险，已知问题是 2026 Q2 ret60 动量反转和单一因子依赖。
- V9 是 V8 防守改造，但完整回测风险调整收益不如 V8，禁止上线。
- V4 胜率和换手有参考价值，但不能替代 V8。
- 财报错杀回归必须接真实财报公告日期后再验证。

## 常用代码位置

- 研究策略目录：`backend/app/services/research_backtest.py`
- V8 策略：`backend/app/services/alpha191_v8.py`
- V9 策略：`backend/app/services/alpha191_v9.py`
- Alpha191 研究策略：`backend/app/services/alpha191_research.py`
- 回测任务入口：`backend/app/tasks.py`
- API 路由：`backend/app/api/routes.py`
- 前端策略实验室：`frontend/components/ResearchLab.tsx`
- 前端外壳：`frontend/components/AppShell.tsx`
- 报告目录：`reports/`

## 因子开发规则

新增因子必须说明：

- 因子名称，使用中文。
- 数据字段来源。
- 是否使用未来数据。
- 是否是 proxy。
- 适用行情。
- 失效场景。
- 回测区间。
- IC / RankIC / 分组收益 / 回撤。

如果使用财务或公告类因子，必须接真实公告日期，不能只用季度月份 proxy。

## 版本命名

新策略用中文命名，保留机器 ID。

示例：

- `V8 动量风险削减版` / `alpha191_v8_risk_reduced`
- `残差超跌回归 V3` / `mr_residual_reversal_v3`
- `筹码套牢释放 V2` / `mr_overhang_release_v2`
- `流动性温和量价 V2` / `liquidity_wq_decorrelated_v2`

## 最小工作流

1. 先读相关代码和报告。
2. 只改必要文件。
3. 本地跑最小验证。
4. 把结果写入 `reports/`。
5. 明确说明：可研究、可模拟、禁止上线，或需要继续验证。

## 安全默认值

任何不确定的动作按“不上线、不删除、不覆盖数据、不碰服务器”处理。
