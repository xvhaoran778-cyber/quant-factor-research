# 量化因子挖掘与多因子策略研究

本仓库用于保存当前阶段表现较好的量化因子、策略脚本和研究报告。研究目标是把项目从零散策略收敛成可复验的研究流水线，并优先复用成熟开源框架的数据与研究能力。

## 当前研究主线

- A股多因子组合：围绕反转、低波动、流动性、资金强度、趋势确认等因子做组合检验。
- 长周期验证：重点比较 2012-2023 与 2024-2026 的收益差异，避免只看近年行情。
- 回撤诊断：分析负收益窗口、风格失效和市场环境变化对策略的影响。
- 攻防切换实验：验证大盘赚钱效应较好时使用进攻型因子、转弱时使用防守型因子的思路。

## 重点文件

- `backend/app/services/new_factors_fast.py`：当前新增和优化后的快速因子计算。
- `scripts/run_multifactor_combination.py`：多因子组合回测入口。
- `scripts/mine_old_period_factors.py`：旧周期因子挖掘，用于解释 2012-2023 表现不足的问题。
- `scripts/diagnose_2012_2026_strategy.py`：2012-2026 长周期诊断。
- `scripts/optimize_2012_2026_multifactor.py`：长周期多因子优化。
- `scripts/analyze_return_strategy.py`：收益型策略回撤分析。
- `scripts/run_regime_switch_strategy.py`：攻防因子切换实验。

## 研究报告

- `reports/old_period_factor_mining_report.md`
- `reports/multifactor_long_validation_report.md`
- `reports/strategy_2012_2026_diagnosis_report.md`
- `reports/regime_switch_strategy_report.md`
- `reports/return_strategy_drawdown_analysis.md`

## 当前结论

近年表现较好的因子不代表长期稳定有效。当前更值得保留的是：

- 经过长周期复验的多因子组合；
- 能解释 2012-2023 失效原因的诊断报告；
- 对 2024-2026 高收益来源有明确归因的策略版本；
- 有回撤分析和市场环境约束的策略，而不是单纯追求历史收益最高的参数。

## 风险提示

本项目仅用于量化研究和回测复验，不构成投资建议。历史收益不能保证未来收益，免费行情数据也可能存在缺失、复权误差和幸存者偏差。实盘前必须继续进行样本外验证、模拟盘跟踪和数据源交叉检查。
