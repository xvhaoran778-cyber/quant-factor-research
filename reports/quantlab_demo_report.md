# QuantLab 演示报告

## 项目概述
QuantLab 是个人从零搭建的全栈 A 股量化投研系统，覆盖数据、因子、策略、回测、AI 决策、模拟交易完整链路。

## 核心成果
- V8 标准版：2012-2026 累计 +7,350%，夏普 1.23
- 30+ 注册策略，29 个 PRESETS 一键可跑
- 6 个 LLM Agent + 1 CIO Coordinator 决策系统
- 8 维数据审计，13,690,824 行零错误

## 最近修复
- 停牌检测：原始数据 `suspended` 列恒为 False，改为平线检测 (open=close=high=low)
- CLI 工具：新增 `cli.py` 终端演示程序

## 使用方式
```bash
python cli.py info
python cli.py run v8 --quick
python cli.py compare
python cli.py agent 600519.SH
```
